# Paper-to-code traceability

| Manuscript item | Implemented in | Verification |
| --- | --- | --- |
| Language-enabled stochastic game (Eq. 1) | `types.py`, `envs/base.py` | environment contract tests |
| Memory tuple and prospective organization | `memory.py::MemoryItem`, `RegimeAwareMemoryBank.add` | `test_memory.py` |
| Regime-aware score (Eq. 2) | `memory.py::retrieve` | matching-regime test |
| Retrospective memory removal | `memory.py::apply_removal_attribution`, trainer local surrogate | reliability-update test |
| Reasoning mode (Eq. 3) | `reasoning.py::ReasoningScheduler` | long/shock/inactive tests |
| Structured claim `Z` | `types.py::StructuredClaim` | claim validation and communication tests |
| Bait, Switch, Edge audit | `communication.py::CounterfactualValues`, `CounterfactualCredibility.audit` | explicit harmful rollout test |
| Message quality (Eq. 4) | `communication.py::quality/select` | deterministic candidate selection path |
| Credibility-calibrated posterior (Eq. 5) | `beliefs.py::update_from_message` | zero/high-credibility tests |
| Sampled deviation loss (Eq. 6) | `losses.py::deviation_regularizer`, `trainer.py::update` | tensor shape checks at runtime |
| Trust-weighted pooling (Eq. 7) | `fusion.py::trusted_message_pool` | end-to-end feature test |
| Centralized TD value loss (Eq. 8) | `losses.py::value_loss`, `trainer.py::update` | training smoke when PyTorch is installed |
| Language credit (Eq. 9) | `language_credit.py::CentralizedLanguageCritic` | completed-trajectory-only API |
| Textual revision (Eq. 10) | `language_credit.py::TextualPolicyReviser` | batched policy store |
| Fast objective (Eq. 11) | `losses.py`, `trainer.py::update` | resolved loss weights logged |
| THMR (Eq. 12) | `metrics.py::trusted_harmful_message_rate` | metric tests |
| Useful@K (Eq. 13) | `metrics.py::useful_at_k` | metric tests |
| Algorithm 1 | `pipeline.py`, `trainer.py::train` | offline integration smoke |

## Deliberate reconstruction choices

The supplied manuscript does not specify all implementation details. The
repository therefore makes the following choices visible instead of hiding
them:

- the exact numerical mapping from Bait/Switch/Edge to strategic risk;
- message-quality and fast-loss coefficients;
- trust threshold, deviation tolerance, gradient frequency, and clipping;
- C1 corruption severity and C3 profile identities;
- prompt text and consolidation rules;
- a deterministic projected BGE representation until learned projection weights
  are released;
- a proxy removal attribution only in smoke mode.

Each is marked `implementation_default` in `configs/base.yaml`. Replacing one
changes the resolved configuration hash.
