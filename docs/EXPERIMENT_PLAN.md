# Experiment plan

This document defines the comparison before implementation begins. Frozen values must not be tuned independently for each experiment.

## 1. Environment contract

The first smoke test must record and validate:

| Item | Initial decision |
| --- | --- |
| Environment | `PushCube-v1` |
| Robot | Panda |
| Observation mode | `state` |
| Control mode | `pd_joint_delta_pos` for the official baseline; `pd_ee_delta_pose` is a later comparison |
| Reward mode | `normalized_dense` for the official baseline |
| Maximum episode steps | 50 |
| Training backend | Linux/NVIDIA GPU for full runs |
| Development backend | Linux/NVIDIA CUDA; CPU only for fixed evaluation and smoke checks |
| ManiSkill version | 3.0.1 |
| Python version | 3.12 |

The official state-based baseline uses `pd_joint_delta_pos` through the default in `ppo_fast.py`. Any controller change is a separate documented comparison.

## 2. Experimental conditions

### A — Easy baseline

- Normalized dense ManiSkill reward.
- Reduced cube and/or robot initial-state randomization.
- Purpose: verify that the full training and evaluation pipeline can learn the task.

### B — Standard task

- Normalized dense ManiSkill reward.
- Default task randomization.
- Purpose: establish the primary reference result.

### C — Reward ablation

- Default task randomization.
- Sparse reward or one explicitly documented modification of the dense reward.
- Purpose: isolate the effect of reward information.

The curriculum comparison is a later phase and must use condition B as its direct-training control.

## 3. Evaluation protocol

- Use the official training seeds `9351`, `4796`, and `1788`.
- Use a fixed, separate set of evaluation seeds.
- Evaluate deterministic actions at a fixed environment-step interval.
- Use the same training-step budget and evaluation episode count for every condition.
- Report the mean and variability across seeds; do not report only the best checkpoint.
- Evaluate generalization on a wider initial-position distribution only after standard evaluation is frozen.

## 4. Metrics

Each evaluation should record:

- success rate;
- undiscounted episode return;
- episode length and steps to success;
- final planar cube-to-goal distance;
- total environment interactions;
- wall-clock training time and environment throughput;
- seed, configuration, checkpoint step, code revision, and dependency versions.

Primary metric: success rate. Primary sample-efficiency measure: environment steps required to reach a fixed success threshold. Return is diagnostic because reward scales differ across reward ablations.

## 5. Planned evidence

- Learning curves with aggregate statistics across seeds.
- A final metrics table for A, B, and C.
- Videos from a random policy, an early checkpoint, and the final policy using fixed evaluation seeds.
- A direct comparison between the official baseline and the custom PPO under a shared protocol.
- A short written discussion of failures and deviations from the original plan.

## 6. Phase exit criteria

### Phase 0 — Environment validation

- [x] Reset and step work on the local CPU backend.
- [x] Observation `(1, 35)` and action shapes are documented: `(8,)` for the official joint controller and `(7,)` for the Cartesian smoke test.
- [x] Seeded resets are checked for repeatability.
- [x] Success, reward, truncation, and final-distance signals are exposed by the evaluation pipeline.
- [x] A random-policy video can be generated.

### Phase 1 — Official baseline

- [x] The exact upstream baseline revision and command are recorded.
- [x] A reduced CPU run validates training, checkpointing, evaluation, and video end to end.
- [x] The 50-million-step CUDA reference run for seed `9351` completes and reaches at least 90% deterministic-evaluation success.
- [ ] Aggregate results across three official seeds are available when final results are assembled.
- [x] A checkpoint produced by the upstream trainer can be loaded by the fixed evaluator.

The authoritative implementation status and sequencing are in [ROADMAP.md](../ROADMAP.md).

### Phase 2 — Custom PPO

- Core PPO calculations have focused tests.
- A short run improves over random behavior.
- The shared evaluation pipeline can evaluate both implementations.
- The comparison uses matched environment interactions and seeds.

### Phase 3 — Ablations and curriculum

- Conditions A, B, and C run over at least three seeds.
- The curriculum rule and progression thresholds are fixed before the comparison.
- Results include aggregate plots, the final table, and representative videos.

## 7. Non-goals for the first iteration

- RGB or RGB-D policies.
- Grasping, stacking, or insertion tasks.
- Distributed training across machines.
- Hyperparameter sweeps before the official baseline is reproduced.
- Changes to PPO architecture during the environment/reward ablations.
