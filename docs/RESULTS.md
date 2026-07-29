# 3M custom PPO reward-alignment pilot

This pilot compares two custom PPO runs on `PushCube-v1` using seed `9351` and the
same fixed deterministic 20-episode evaluator. Both use 4,096 CUDA environments,
four rollout steps, entropy coefficient `0.005`, four update epochs, and linear
learning-rate decay. Checkpoints are evaluated every 500,000 interactions and the
best one is selected by success rate, then final distance, then return.

| Measure | Stabilized dense reward | Dense reward minus 0.01/step |
| --- | ---: | ---: |
| Requested / executed interactions | 3,000,000 / 3,014,656 | 3,000,000 / 3,014,656 |
| Best selected success rate | 85% (17/20) | 85% (17/20) |
| Final checkpoint success rate | 65% | 85% |
| Best-policy mean episode length | 20.8 | 33.6 |
| Best-policy mean steps to success | 15.6 | 30.7 |
| Best-policy final cube-to-goal distance | 0.1026 m | 0.1006 m |
| Training throughput | 19,032 steps/s | 18,490 steps/s |

![Checkpoint comparison](images/custom_ppo_time_penalty_3m.png)

## Interpretation

The per-step cost is intended to make reward favour task completion without
dithering. In this pilot, it does not improve the peak success rate: both conditions
select a policy with 17 successes out of 20. It also does not yet meet the
fewer-steps objective: the control policy reaches success faster on this evaluator.

Its useful signal is late-run stability. From 1.5M interactions onward, the
time-cost run stays between 75% and 85% success, whereas the control drops to 35%
at 2M and finishes at 65%. This is one stochastic training seed, so it is evidence
for a follow-up—not a general conclusion. The primary 50M reference should remain
the standard dense-reward configuration; repeat this ablation for seeds `9351`,
`4796`, and `1788` before treating it as an improvement.

The evaluated local videos are:

- `artifacts/evaluations/custom-ppo-cuda-stabilized-3M-control-best/videos/0.mp4`
- `artifacts/evaluations/custom-ppo-cuda-time-penalty-3M-best/videos/0.mp4`

Those are generated artifacts and remain outside Git. Regenerate them with
`scripts/evaluate.py` without `--no-video`.
