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
| Custom PPO | CUDA baseline ready | The trainer now uses bounded Gaussian actions and vectorized CUDA rollouts. A 4,096-environment smoke run completes at 2,403 interactions/s; the 50M matched-budget run is ready to launch. |
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

## Completed diagnosis — 50k CPU custom PPO does not demonstrate task completion

Run the custom PPO for 50,000 CPU interactions with the frozen configuration, then evaluate it with the shared 20-seed protocol.

```bash
uv run scripts/train_custom_ppo.py \
  --total-timesteps 50000 \
  --output-dir runs/custom-ppo-cpu-50k
uv run scripts/evaluate.py runs/custom-ppo-cpu-50k/final_ckpt.pt
```

- [x] Record the success rate, return, and final cube-to-goal distance. With bounded actions, the 20-seed return rises from 2.07 to 7.60; final distance is 0.2044 and success remains 0/20.
- [x] Diagnose the absence of success. The prior trainer evaluated log-probabilities for unclipped Gaussian samples while the environment received clamped actions; this is fixed with a `tanh` action transform. The remaining 50k run is still only 0.1% of the 50M official budget, so it cannot establish a learning baseline.

## Next milestone — matched-budget custom PPO comparison

Scale the custom implementation to the same 50-million-interaction budget and fixed 20-seed evaluator as the official baseline. The vectorized CUDA trainer is ready; the single-environment CPU implementation remains the integration-test path.

- [x] Implement vectorized CUDA rollout collection and validate a `4,096 × 4` rollout on the RTX 3090 (2,403 interactions/s).
- [ ] Train and evaluate seed `9351` for 50 million interactions with the shared protocol.
- [ ] Confirm at least one successful fixed-seed episode before starting ablations.

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
