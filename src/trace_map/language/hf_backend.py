"""Optional Hugging Face implementations for Qwen3 and BGE-M3."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from trace_map.language.base import GeneratedStatement, MessageRequest, ReasoningRequest
from trace_map.types import StructuredClaim, to_jsonable


class SentenceTransformerTextEncoder:
    """Frozen BGE encoder with a deterministic projection to the paper's 64-D space.

    The original manuscript calls the projection trainable but does not provide
    its checkpoint. The deterministic projection is an explicit reconstruction
    default; replace ``projection`` with the released learned weights for exact
    numerical reproduction.
    """

    def __init__(self, config: dict[str, Any]):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError("Install language dependencies with: pip install -e '.[language]'") from error
        self.model = SentenceTransformer(
            config["encoder_model"],
            revision=config.get("encoder_revision"),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
        )
        self._dimension = int(config["embedding_dim"])
        raw_dimension = int(self.model.get_sentence_embedding_dimension())
        rng = np.random.default_rng(0)
        projection = rng.normal(0.0, 1.0 / np.sqrt(raw_dimension), (raw_dimension, self._dimension))
        self.projection = projection.astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        raw = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        projected = np.asarray(raw, dtype=np.float32) @ self.projection
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        return projected / np.maximum(norms, 1e-12)


class HuggingFaceLanguagePolicy:
    def __init__(self, config: dict[str, Any]):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise ImportError("Install language dependencies with: pip install -e '.[language,train]'") from error
        self.torch = torch
        self.config = config
        prompt_file = config.get("prompt_file")
        if prompt_file:
            with Path(prompt_file).open("r", encoding="utf-8") as handle:
                self.prompts = yaml.safe_load(handle)
        else:
            prompt_resource = resources.files("trace_map").joinpath("prompts.yaml")
            with prompt_resource.open("r", encoding="utf-8") as handle:
                self.prompts = yaml.safe_load(handle)
        model_name = config["generator_model"]
        revision = config.get("generator_revision")
        trust_remote_code = bool(config.get("trust_remote_code", False))
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()

    def _complete(
        self,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        sampling = temperature > 0
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_tokens,
                do_sample=sampling,
                temperature=temperature if sampling else None,
                top_p=top_p if sampling else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    @staticmethod
    def _json_object(text: str) -> dict[str, Any]:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"Language model did not return a JSON object: {text[:200]}")
        payload = json.loads(text[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        return payload

    def reason(self, request: ReasoningRequest) -> str:
        user = json.dumps(to_jsonable(request.__dict__), sort_keys=True)
        return self._complete(
            self.prompts["reasoning"],
            user,
            request.max_tokens,
            temperature=0.0,
        )

    def generate_statements(self, request: MessageRequest, count: int) -> list[GeneratedStatement]:
        statements: list[GeneratedStatement] = []
        for index in range(count):
            generation_prompt = {
                **to_jsonable(request.__dict__),
                "candidate_index": index,
                "instruction": "Write one concise public statement; do not reveal private state.",
            }
            statement_text = self._complete(
                self.prompts["message_generation"],
                json.dumps(generation_prompt, sort_keys=True),
                request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
            extraction_prompt = {
                "statement": statement_text,
                "economic_context": {
                    "public": request.public,
                    "previous_public": request.previous_public,
                    "regime": to_jsonable(request.regime),
                },
                "schema": {
                    "variable": "string",
                    "direction": "increase|decrease|stable|unknown",
                    "horizon": "positive integer",
                    "sender_commitment": f"array length {request.sender_action_dim} or null",
                    "receiver_recommendation": f"array length {request.receiver_action_dim} or null",
                    "extraction_confidence": "number in [0,1]",
                },
            }
            raw = self._complete(
                self.prompts["claim_extraction"],
                json.dumps(extraction_prompt, sort_keys=True),
                max_tokens=192,
                temperature=0.0,
            )
            payload = self._json_object(raw)
            sender = payload.get("sender_commitment")
            receiver = payload.get("receiver_recommendation")
            claim = StructuredClaim(
                variable=str(payload.get("variable", "unknown")),
                direction=str(payload.get("direction", "unknown")).lower(),
                horizon=max(1, int(payload.get("horizon", request.horizon))),
                sender_commitment=None if sender is None else np.asarray(sender, dtype=np.float32),
                receiver_recommendation=None
                if receiver is None
                else np.asarray(receiver, dtype=np.float32),
                extraction_confidence=float(payload.get("extraction_confidence", 0.0)),
            )
            claim.validate()
            statements.append(GeneratedStatement(statement_text, claim))
        return statements

    def propose_revision(
        self, textual_policy: str, trajectory: list[dict[str, Any]], credits: dict[str, float]
    ) -> str:
        payload = {"textual_policy": textual_policy, "trajectory": trajectory, "credits": credits}
        return self._complete(
            self.prompts["language_revision"],
            json.dumps(to_jsonable(payload), sort_keys=True),
            max_tokens=256,
            temperature=0.0,
        )

    def consolidate_revisions(self, textual_policy: str, proposals: list[str]) -> str:
        payload = {"current_policy": textual_policy, "proposals": proposals}
        return self._complete(
            self.prompts["consolidation"],
            json.dumps(payload, sort_keys=True),
            max_tokens=384,
            temperature=0.0,
        )
