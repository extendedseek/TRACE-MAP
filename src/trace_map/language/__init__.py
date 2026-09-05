"""Language-policy and text-encoder factories."""

from __future__ import annotations

from typing import Any

from trace_map.language.base import LanguagePolicy, TextEncoder


def make_language_components(config: dict[str, Any]) -> tuple[LanguagePolicy, TextEncoder]:
    language_cfg = config["language"]
    generator_backend = language_cfg["generator_backend"]
    encoder_backend = language_cfg["encoder_backend"]

    if generator_backend == "template":
        from trace_map.language.template_backend import TemplateLanguagePolicy

        policy: LanguagePolicy = TemplateLanguagePolicy()
    elif generator_backend == "huggingface":
        from trace_map.language.hf_backend import HuggingFaceLanguagePolicy

        policy = HuggingFaceLanguagePolicy(language_cfg)
    else:
        raise ValueError(f"Unsupported generator backend: {generator_backend}")

    if encoder_backend == "hashing":
        from trace_map.language.template_backend import HashingTextEncoder

        encoder: TextEncoder = HashingTextEncoder(int(language_cfg["embedding_dim"]))
    elif encoder_backend == "sentence_transformers":
        from trace_map.language.hf_backend import SentenceTransformerTextEncoder

        encoder = SentenceTransformerTextEncoder(language_cfg)
    else:
        raise ValueError(f"Unsupported encoder backend: {encoder_backend}")
    return policy, encoder


__all__ = ["LanguagePolicy", "TextEncoder", "make_language_components"]
