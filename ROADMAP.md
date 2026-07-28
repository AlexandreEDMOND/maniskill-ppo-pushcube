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
| Custom PPO | Short validation complete | The 5,000-step CPU run raises five-seed mean return from 1.90 to 6.64. The shared 20-seed evaluator loads its checkpoint and records a video, but task success remains 0%. |
| Reward/randomization experiments | Not started | Protocol is defined in `docs/EXPERIMENT_PLAN.md`. |
| Curriculum comparison | Not started | Depends on a stable standard-task implementation. |

Experiment outputs remain ignored by Git. The `9351` artifacts above are therefore local evidence, not versioned project files.

## Completed milestone — minimal custom PPO

Implement a small state-based PPO trainer that uses the same task contract and evaluation pipeline as the official reference.

- [x] Add a minimal actor-critic MLP.
- [x] Add rollout collection, GAE, clipped policy objective, value loss, and entropy bonus.
- [x] Add focused tests for the PPO calculations.
- [x] Verify that a 5,000-step CPU run improves deterministic mean return over the initial policy (1.90 to 6.64 on five fixed seeds).
- [x] Evaluate the custom checkpoint through `scripts/evaluate.py` on 20 fixed seeds and record a video.

The short run is an integration check, not a competitive result: its 20-seed success rate remains 0%. The first full comparison must use the same environment-step budget and evaluation protocol as the official baseline.

## Next milestone — demonstrate task completion

Run the custom PPO for 50,000 CPU interactions with the frozen configuration, then evaluate it with the shared 20-seed protocol.

```bash
uv run scripts/train_custom_ppo.py \
  --total-timesteps 50000 \
  --output-dir runs/custom-ppo-cpu-50k
uv run scripts/evaluate.py runs/custom-ppo-cpu-50k/final_ckpt.pt
```

- [ ] Record the success rate, return, and final cube-to-goal distance against the 5,000-step check.
- [ ] Confirm that the longer run produces at least one successful fixed-seed episode before treating it as a learning baseline.
- [ ] If it does not, diagnose the training behavior before changing the frozen experiment conditions.

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
