# Experiment protocol

## Evaluation questions

- **EQ1:** household return, government return, social welfare, and stress
  robustness against numerical and language-guided baselines.
- **EQ2:** probabilistic trust quality under natural, factually corrupted, and
  strategically deviating messages.
- **EQ3:** behavioral-profile inference and empirical exploitability.
- **EQ4:** memory usefulness, language-credit alignment, and regime-shift
  recovery.
- **EQ5:** decision latency and scaling with household count.

## Scenario matrix

Main economic evaluation uses E1, E2, and E3. Communication evaluation uses C0,
C1, C2 at deviation rates 0.10/0.30/0.50, and C3 persistent profiles. Memory
evaluation uses E1→E2 and E1→E3 at period 150 without resetting memory.

Every reported condition should use five seeds. The manuscript states the seed
count but not their identities, so the repository default `[0,1,2,3,4]` is an
implementation choice.

## Training phases

1. Collect 10,000 exploratory transitions.
2. Continue to 300,000 environment steps.
3. Update value models and actors from replay.
4. Distill local value surrogates and apply the sampled-deviation penalty.
5. Update memory reliability from critic-based removals.
6. Every 12 completed episodes, consolidate trajectory-level language revisions.
7. Freeze a checkpoint before evaluating conditions and perturbations.

## Required run artifacts

Each run directory contains:

- `resolved_config.yaml` and its SHA-256 digest;
- `run_metadata.json` with Python/PyTorch/CUDA/hardware information;
- `training.jsonl`, `episodes.jsonl`, and auditable decision traces;
- checkpoints with a compatibility digest;
- `metrics.json` generated from evaluation logs.

## Fair comparison requirements

Baselines must use the same TaxAI state/action interface, interaction budget,
scenario interventions, and five evaluation seeds. Language-enabled baselines
must use the same Qwen backbone and generation budget unless their method
requires a distinct optimizer. Trust scores that are not probabilities must be
calibrated on validation environments before Brier evaluation.

Do not tune coefficients on final E1/E2/E3, C0–C3, or shift evaluations. The
original validation environments and chosen coefficients were not included in
the supplied manuscript and must be released for Level-3 reproduction.
