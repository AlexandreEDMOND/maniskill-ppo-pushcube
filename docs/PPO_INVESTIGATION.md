# PPO investigation log

This is a running scientific notebook for the custom PPO implementation. It
records hypotheses, controlled changes, observations, and the strength of the
result. Checkpoint success is always measured on the same deterministic 20 CPU
evaluation seeds. One training seed is a pilot result, not a final claim.

## Measurement rules

| Quantity | How to interpret it |
| --- | --- |
| Success rate | Primary outcome. Higher is better; with 20 episodes, one episode equals 5 percentage points. |
| Final cube-to-goal distance | Secondary completion signal. Lower is better; the task-success boundary is approximately 0.1 m. |
| Raw return | Diagnostic only. It can rise without success because `normalized_dense` rewards progress and can reward a long near-goal trajectory. |
| Steps to success | Efficiency metric among successful episodes. Lower is better. |
| Action sigma | Gaussian policy exploration scale. A gradual reduction is expected; an abrupt collapse toward zero is a warning. |
| Approximate KL / clip fraction | PPO update-size diagnostics. Persistently large values mean updates are too aggressive. |

The checkpoint selector ranks success first, lower final distance second, and
return third. Final checkpoints must not be substituted for the selected
checkpoint.

## Completed investigations

| ID | Hypothesis / change | Evidence | Observation | Current conclusion |
| --- | --- | --- | --- | --- |
| I1 | A 50k CPU run can validate learning. | 50k interactions, 20 fixed seeds. | Return rose to 7.60 but success remained 0%; final distance 0.2044 m. | Integration evidence only; 50k is far too small for a performance claim. |
| I2 | Bounded actions must use a matching policy likelihood. | Custom PPO action transform fix. | The trainer previously scored an unclipped Gaussian action while the environment received a clipped one. | Fixed with a tanh-squashed Gaussian; this is a correctness fix, not a tuning result. |
| I3 | Vectorized CUDA collection makes matched-scale experiments practical. | 4,096 environments x 4 steps. | Throughput is about 17k–19k steps/s; a 5M pilot takes about five minutes. | CUDA is the validated training path. |
| I4 | Periodic fixed-seed selection protects against late policy degradation. | 5M entropy-0.005 pilot. | Best checkpoint at 508k: 80% success; final checkpoint at 5M: 0%. | Keep and report `best_ckpt.pt`, never assume the final checkpoint is best. |
| I5 | More exploration helps discover success. | Entropy coefficient increased from 0.001 to 0.005. | Peak selected success improved in the pilot, but later degradation remained. | Keep 0.005 for now; it is not sufficient to guarantee stability. |
| I6 | Smaller PPO updates should reduce destructive policy drift. | Standard dense 3M pilot: 4 epochs, linear LR decay. | Best: 85% at 1.016M; final: 65%. A drop to 35% occurred at 2.015M. Sigma declined smoothly from 0.91 to 0.66. | Better than the earlier 5M final collapse, but still not stable enough for a final claim. |
| I7 | A per-step cost aligns reward with quick completion. | 3M matched pilot, reward minus 0.01 per step. | Peak success ties control at 85%; final success is 85% versus 65%. But steps to success are 30.7 versus 15.6 for control. | Possible late-run stability benefit; it does not yet improve efficiency. Replicate over three seeds before use. |
| I8 | A conservative package of smaller PPO updates prevents late degradation. | 5M dense pilot: LR `1e-4`, target KL `0.03`, 2 epochs; other settings unchanged. | Success reaches 95% at 3.015M and stays 95% through 5.014M. Independent evaluation of the 4.506M selected checkpoint: 19/20 success, 13.1 steps to success, 0.0948 m final distance. KL declines 0.0106 to 0.0023; sigma 0.983 to 0.907. | Strong one-seed evidence for the combined package. It cannot identify which lever caused the improvement and requires seeds `4796` and `1788`. |

## Conservative PPO update details

**Question tested.** Does reducing the initial update scale prevent the late
degradation seen in the 3M dense-control pilot?

**Controlled configuration.** This 5M run keeps the environment, seed, network,
entropy coefficient, rollout, evaluation seeds, and LR decay. It changes three
coupled update-size controls at once:

| Parameter | Dense-control pilot | Conservative 5M test |
| --- | ---: | ---: |
| Initial learning rate | 3e-4 | 1e-4 |
| `target_kl` | 0.10 | 0.03 |
| Update epochs | 4 | 2 |

Because three levers change together, the result answers whether the conservative
package is better; it cannot attribute an effect to one parameter. If it remains
stable across seeds, the next experiment should ablate one lever at a time.

## Next scientific steps

- Repeat the 5M conservative setting for seeds `4796` and `1788`, then report the
  mean and standard deviation of success, distance, and steps to success.
- Use the conservative dense-reward setting, with periodic best-checkpoint
  selection, for the next 50M reference candidate. Do not use the final checkpoint
  by default.
- Keep the dense reward as the reference unless the time-cost reward improves both
  success stability and steps to success across three seeds.
- To identify a cause rather than a package effect, vary one of learning rate,
  target KL, or update epochs at a time on repeated seeds.
