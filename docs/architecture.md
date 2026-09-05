# Architecture and information flow

TRACE-MAP separates decentralized execution from centralized training. The
boundary is enforced by data structures: `ObservationBundle.local` and
`ObservationBundle.public` are actor-facing, while the global state and
counterfactual targets enter only critics, training losses, or evaluation.

## Execution path

1. **Observe.** The government receives aggregate variables; household `i`
   receives the same aggregate vector plus only its own assets and productivity.
2. **Retrieve.** A dense semantic shortlist is reranked by regime compatibility,
   learned reliability, and age. Training can use Gumbel top-k; evaluation uses
   deterministic top-k.
3. **Reason.** Long reasoning occurs at configured checkpoints. Short reasoning
   occurs when any monitored indicator exceeds its calibrated shock threshold.
4. **Communicate.** Activated agents generate candidates paired with a variable,
   direction, horizon, sender commitment, receiver recommendation, and extraction
   confidence.
5. **Verify.** Each sender–receiver pair receives a factual score, commitment
   consistency, Bait/Switch/Edge values, strategic risk, influence, and trust.
6. **Infer.** The receiver's profile posterior is updated with a likelihood mixed
   toward an uninformative likelihood in proportion to low credibility. Observed
   public behavior then refines the posterior.
7. **Fuse and act.** Trusted message embeddings are normalized by their trust
   weights and concatenated with local observation, reasoning, memory, and belief
   representations. The decentralized actor emits a bounded physical action.
8. **Attribute and update.** Memory-removal values update reliability. Replay
   transitions update actors/critics frequently; completed trajectories revise
   textual policies every `U_text` episodes.

## Training-only information

The centralized critic receives the global state, joint actions, and all agents'
language contexts. It supplies temporal-difference targets, sampled-deviation
values, memory-removal values, and local-surrogate targets. The language critic
can inspect a completed trajectory, individual return, and social return.

The following are never supplied to action selection:

- other households' private asset or productivity state;
- ground-truth persistent reliability/profile labels;
- future sender actions or future economic outcomes;
- harmful-message labels and counterfactual return targets;
- centralized critic or language-critic hidden state.

## Runtime backends

| Layer | Smoke/CI | Paper-scale path |
| --- | --- | --- |
| Economy | `MockEconomicSociety` | pinned TaxAI `economic_society` adapter |
| Text generation | deterministic templates | Qwen3-32B through Transformers |
| Text encoding | signed feature hashing | frozen BGE-M3 plus 64-D projection |
| Physical policy | deterministic heuristic | per-agent PyTorch actors |
| Value model | proxy removal values | centralized critics and distilled local surrogates |

The deterministic BGE projection is only a reconstruction default because the
paper states that the projection is trainable but does not supply its learned
weights. Exact reproduction requires replacing it with the released projection
checkpoint.

## Computational scaling

Dense candidate verification has cost `O(H²MR)` for `H` households, `M`
candidates, and `R` sampled deviations. `communication.retained_senders`
implements the paper's relevant-sender restriction, reducing the dominant term
to `O(HSMR)` when `S << H`.
