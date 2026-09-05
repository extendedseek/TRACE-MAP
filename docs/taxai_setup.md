# TaxAI integration

TRACE-MAP targets upstream TaxAI revision
`04e7cb17071d942366eb0cad4fb4ba57d02bf612`.

The adapter uses the public interface shown by upstream:

```python
from env.env_core import economic_society
from omegaconf import OmegaConf

cfg = OmegaConf.load("cfg/default.yaml")
env = economic_society(cfg.Environment)
global_obs, private_obs = env.reset()
next_global, next_private, gov_reward, household_reward, done = env.step(actions)
```

## Role mapping

| TRACE-MAP ID | TaxAI entity | Observation | Action |
| --- | --- | --- | --- |
| `government` | `government` | seven aggregate statistics | five bounded tax/expenditure controls |
| `household_i` | row `i` of `Household` | aggregate vector + own productivity/assets | saving and labor |

The adapter stacks household actions back into TaxAI's `(H, 2)` array. It does
not alter TaxAI transition or reward functions. Language affects only the
TRACE-MAP policy that selects these physical actions.

## Stress interventions

| Condition | Depreciation | Consumption tax | Interest |
| --- | ---: | ---: | ---: |
| E1 low stress | 0.055 | 0.055 | 0.035 |
| E2 financial tightening | 0.080 | 0.045 | 0.070 |
| E3 compound stress | 0.110 | 0.090 | 0.085 |

For an E1→E2/E3 run, the adapter updates only these three simulator attributes at
period 150 and preserves the TRACE-MAP memory banks.

## Installation caveat

At the pinned revision, TaxAI reads `agents/data/advanced_scfp2022.csv` through a
relative path. The adapter temporarily enters the checkout directory for
construction, reset, and step calls, then restores the caller's working
directory.

The upstream repository did not expose an explicit license when this package
was assembled. For that reason TaxAI is not copied into this archive. Confirm
the upstream terms before redistribution.
