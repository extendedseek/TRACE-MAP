# Language component notes

## Qwen3-32B generator

The paper uses Qwen3-32B for private reasoning, candidate statements, structured
claim extraction, factual verification, and language-credit assessment.
Generation uses temperature 0.6 and top-p 0.9 for candidate diversity, while
extraction, verification, and credit tasks are deterministic. Reasoning and
messages are capped at 256 and 96 tokens.

The repository's Hugging Face adapter uses the model's chat template. Pin a
specific model revision before a numerical-reproduction claim. Hardware needs
depend on precision and serving strategy; 32B inference ordinarily requires a
multi-GPU, quantized, or remote-serving setup.

## BGE-M3 encoder

BGE-M3 is frozen. The paper maps embeddings to 64 dimensions through a trainable
projection. Because that projection checkpoint is absent, the repository uses a
seeded deterministic projection for reconstruction. It is shape-compatible but
not numerically equivalent to the learned projection.

## Offline fallback

The hashing encoder and template policy are deliberately simple and fully
auditable. They validate control flow, privacy boundaries, logging, and metric
calculation. Results produced with them must be labeled `smoke`, never TRACE-MAP
paper reproduction.

## Model safety

Prompts prohibit disclosure of another household's private state and require
structured actions in `[-1,1]`. Leave `trust_remote_code` disabled unless the
specific model repository has been reviewed. Never commit authentication tokens
or proprietary model responses.
