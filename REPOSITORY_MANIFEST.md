# Repository manifest

This inventory answers which files are required for a reviewable and
reproducible TRACE-MAP release. A file is considered required when it provides
an executable method component, pins an experimental choice, validates behavior,
or supplies standard GitHub project governance.

## Core project files

| File | Role |
| --- | --- |
| `README.md` | Installation, method summary, commands, and limitations |
| `REPRODUCIBILITY.md` | Reproduction levels and evidence checklist |
| `REPOSITORY_MANIFEST.md` | Complete repository inventory |
| `pyproject.toml` | Package metadata, dependencies, CLI, and tool configuration |
| `requirements*.txt`, `environment.yml` | pip and Conda environments |
| `Dockerfile` | Portable CPU smoke/test image |
| `Makefile` | Common setup, test, lint, and smoke commands |
| `CITATION.cff` | GitHub citation metadata |
| `LICENSE` | Interim copyright notice; author must select public-release terms |
| `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` | Collaboration and reporting policies |
| `.gitignore`, `.gitattributes`, `.editorconfig`, `.pre-commit-config.yaml` | Repository hygiene |

## Method implementation

| File | Paper construct |
| --- | --- |
| `src/trace_map/config.py` | Validated hierarchical experiment configuration |
| `src/trace_map/types.py` | Observations, transitions, claims, audits, and trace records |
| `src/trace_map/memory.py` | Eq. 2 memory score, top-B/top-K retrieval, EMA reliability, removal attribution |
| `src/trace_map/reasoning.py` | Eq. 3 short/long/inactive reasoning activation |
| `src/trace_map/communication.py` | Structured claims, counterfactual audit, Eq. 4 quality, trust |
| `src/trace_map/beliefs.py` | Eq. 5 credibility-calibrated posterior and behavioral refinement |
| `src/trace_map/fusion.py` | Eq. 7 trusted pooling and local feature construction |
| `src/trace_map/losses.py` | Eqs. 6, 8, and 11 fast-timescale objectives |
| `src/trace_map/language_credit.py` | Eqs. 9–10 trajectory credit and textual revision |
| `src/trace_map/pipeline.py` | End-to-end decentralized inference pipeline |
| `src/trace_map/replay.py` | Fixed-capacity transition replay |
| `src/trace_map/trainer.py` | Two-timescale training loop and checkpointing |
| `src/trace_map/metrics.py` | THMR, Brier, NLL, exploitability, Useful@K, correlation, recovery |
| `src/trace_map/evaluate.py` | Evaluation and regime-shift runner |
| `src/trace_map/cli.py` | User-facing smoke, train, evaluate, and aggregate commands |
| `src/trace_map/envs/base.py` | Environment contract used by the framework |
| `src/trace_map/envs/mock_economy.py` | Fast deterministic integration environment |
| `src/trace_map/envs/taxai_adapter.py` | Pinned TaxAI `economic_society` adapter |
| `src/trace_map/language/base.py` | Generator and text-encoder protocols |
| `src/trace_map/language/template_backend.py` | Offline deterministic language backend |
| `src/trace_map/language/hf_backend.py` | Qwen/BGE Hugging Face adapters |
| `src/trace_map/prompts.yaml` | Versioned reconstruction prompts for Qwen tasks |
| `src/trace_map/models/networks.py` | Actor, centralized critic, belief, credibility, selector, and surrogate networks |

## Experiment specification

| Path | Content |
| --- | --- |
| `configs/base.yaml` | Paper-scale architecture and training defaults |
| `configs/smoke.yaml` | Small dependency-light test setting |
| `configs/environment/e1.yaml` | Low-stress economy: 0.055/0.055/0.035 |
| `configs/environment/e2.yaml` | Financial tightening: 0.080/0.045/0.070 |
| `configs/environment/e3.yaml` | Compound stress: 0.110/0.090/0.085 |
| `configs/communication/c0.yaml` | Natural communication |
| `configs/communication/c1.yaml` | Factual corruption |
| `configs/communication/c2_*.yaml` | Strategic deviation at 0.10/0.30/0.50 |
| `configs/communication/c3.yaml` | Persistent reliability profiles |
| `configs/ablation/*.yaml` | Component switches used by the ablation runner |
| `scripts/reproduce_paper.sh` | Seed/scenario orchestration |
| `scripts/setup_taxai.sh` | Exact upstream revision checkout |
| `results/reference/paper_claims.yaml` | Manuscript headline values, never used as generated output |

## Documentation and validation

| Path | Purpose |
| --- | --- |
| `docs/architecture.md` | Module boundaries and information-isolation rules |
| `docs/paper_to_code.md` | Equation/algorithm-to-source mapping |
| `docs/taxai_setup.md` | Upstream interface and stress-parameter mapping |
| `docs/experiment_protocol.md` | Seeds, phases, scenarios, logging, and comparisons |
| `docs/model_cards.md` | Qwen/BGE use, hardware expectations, and fallback limitations |
| `tests/test_*.py` | Config, memory, reasoning, communication, beliefs, metrics, environment, and smoke tests |
| `.github/workflows/ci.yml` | Automated syntax, tests, packaging, and smoke checks |
| `.github/ISSUE_TEMPLATE/*`, `.github/pull_request_template.md` | Reproducible issue and change reporting |

Large model weights, TaxAI source, generated checkpoints, raw experiment logs,
and submitted manuscript source are intentionally excluded. Their required
locations and provenance are documented rather than hidden in the archive.
