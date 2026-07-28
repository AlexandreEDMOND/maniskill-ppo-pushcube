# Roadmap

This is the authoritative tracker for the project. The experiment plan defines the frozen protocol; this document records what has been completed and what comes next.

## Project goal

Produce a small, reproducible PPO study on ManiSkill's `PushCube-v1`: first establish a reliable reference run, then measure the effect of reward design, initial-state randomization, and curriculum learning.

## Current state

| Area | Status | Evidence |
| --- | --- | --- |
| Environment contract | Complete | CPU smoke test validates seeded reset, signals, time limit, and video recording. |
| Official PPO integration | Complete | The upstream trainer, checkpointing, evaluator, and video pipeline work end to end locally. |
| Official baseline, seed `9351` | Complete | 50 million CUDA interactions; fixed deterministic evaluation: 20/20 successes (100%). Local artifacts: `artifacts/pushcube-9351/` and `artifacts/evaluations/official-9351/`. |
| Official aggregate result | Deferred | Keep `9351` as the reference run now; add other seeds only for the final aggregate report. |
| Custom PPO | Not started | No custom implementation has been added. |
| Reward/randomization experiments | Not started | Protocol is defined in `docs/EXPERIMENT_PLAN.md`. |
| Curriculum comparison | Not started | Depends on a stable standard-task implementation. |

Experiment outputs remain ignored by Git. The `9351` artifacts above are therefore local evidence, not versioned project files.

## Next milestone — minimal custom PPO

Implement a small state-based PPO trainer that uses the same task contract and evaluation pipeline as the official reference.

- [ ] Add a minimal actor-critic MLP.
- [ ] Add rollout collection, GAE, clipped policy objective, value loss, and entropy bonus.
- [ ] Add focused tests for the PPO calculations.
- [ ] Verify that a short training run improves over a random policy.
- [ ] Evaluate the custom checkpoint through `scripts/evaluate.py` with fixed evaluation seeds.

**Exit criterion:** a reproducible short run shows learning and produces the shared evaluation report and video. The first full comparison must use the same environment-step budget and evaluation protocol as the official baseline.

## Experiments after custom PPO

Run each condition with the same interaction budget, evaluation seeds, and reporting protocol. Use at least three training seeds for final comparisons.

| ID | Condition | Purpose |
| --- | --- | --- |
| A | Dense reward with reduced initial-state randomization | Confirm the full training pipeline on an easier setting. |
| B | Dense reward with standard randomization | Establish the primary reference condition. |
| C | Sparse or explicitly modified dense reward with standard randomization | Measure the contribution of reward information. |

For each condition, report success rate, return, episode length, steps to success, final cube-to-goal distance, throughput, wall time, configuration, seed, code revision, and dependency versions. Produce aggregate learning curves, a final mean +/- standard-deviation table, and fixed-seed videos.

## Final baseline aggregate

When results are ready to be finalized, run and evaluate the official seeds `4796` and `1788` alongside the completed `9351` reference run.

- [ ] Complete and evaluate seed `4796`.
- [ ] Complete and evaluate seed `1788`.
- [ ] Aggregate the three official seeds as mean +/- standard deviation.
- [ ] Confirm the fixed evaluator reaches at least 90% success for every seed.

The exact Linux/RTX 3090 procedure is in [docs/TOULOUSE_GPU_RUNBOOK.md](docs/TOULOUSE_GPU_RUNBOOK.md).

## Later milestone — curriculum

Compare direct training on condition B with an adaptive curriculum. Freeze the curriculum rule and progression thresholds before training, then use the same seed count and budget as the direct-training control.

## Out of scope for now

- RGB or RGB-D policies
- Other manipulation tasks, including `PushT-v1`
- Distributed multi-machine training
- Hyperparameter sweeps before the reference comparison is stable

## Completion definition

The first study is complete when:

- the custom PPO and the official baseline are evaluated under a shared protocol;
- conditions A, B, and C have aggregate three-seed results;
- the direct and curriculum variants are compared fairly;
- reproducibility metadata, plots, final tables, and representative videos are available.
