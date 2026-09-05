# Reproducibility statement

TRACE-MAP has three deliberately distinct reproducibility levels.

## Level 1: structural verification

The offline smoke run verifies that every manuscript-defined stage is executed,
that local information is kept separate from training-only state, and that logs
contain the inputs and outputs needed to audit a decision. It does **not** test
the paper's numerical claims.

Required evidence:

- all unit tests pass;
- `trace-map smoke` completes with finite rewards and metrics;
- the decision trace contains retrieval scores, reasoning mode, message audit,
  posterior change, trust weight, action, and attribution;
- repeated runs with the same seed are byte-stable apart from timestamps.

## Level 2: method reimplementation

This level runs the framework in the pinned TaxAI revision using Qwen3-32B and
BGE-M3 with the manuscript hyperparameters and the repository's explicitly
labeled implementation defaults.

Required evidence:

- the exact TaxAI and package revisions are recorded;
- resolved configs and model revisions are stored per run;
- five independent seeds complete for each reported scenario;
- evaluation uses frozen checkpoints and no training-time privileged state;
- generated tables are computed only from JSONL logs.

Results at this level are a faithful reimplementation, but minor differences
from the manuscript may remain because original prompts, checkpoints, and all
validation-selected coefficients were not included in the supplied paper.

## Level 3: numerical reproduction

Claim numerical reproduction only after the original author adds:

1. exact Qwen and BGE model revisions and chat templates;
2. all reasoning, extraction, verification, and language-critic prompts;
3. validation-selected coefficients and trust thresholds;
4. original initial checkpoints or their full training procedure;
5. exact train/validation/evaluation seed sets;
6. raw per-step and per-episode logs for all five seeds;
7. the scripts used to draw every manuscript figure and table.

The targets in `results/reference/paper_claims.yaml` are for comparison only.
They are never read by the trainer or evaluator.

## Determinism

The code seeds Python, NumPy, and PyTorch when installed. Exact GPU
reproducibility still depends on hardware, CUDA, kernel selection, model
quantization, and inference server behavior. Every run records these details in
`run_metadata.json`.

## Information isolation

During action selection, an agent can access only its local observation,
retrieved private memories, its reasoning state, received public messages,
stored opponent beliefs, and quantities computed from those inputs. The global
state, other agents' private observations, ground-truth profile labels, future
actions, harmful-message labels, and counterfactual targets are restricted to
training or evaluation code.
