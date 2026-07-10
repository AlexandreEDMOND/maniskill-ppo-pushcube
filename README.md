# Reproducing PPO for Robotic Object Pushing in ManiSkill

This project studies how reward shaping and initial-state randomization affect PPO on ManiSkill's `PushCube-v1` task. The end goal is a small, reproducible deep reinforcement learning study rather than a one-off training run.

> **Status:** reproducible environment and CPU smoke test are in place. No training implementation has been added yet.

## Local setup

The project uses Python 3.12 and [`uv`](https://docs.astral.sh/uv/). ManiSkill and PyTorch are pinned in `pyproject.toml`, while `uv.lock` records the complete environment.

```bash
uv sync --frozen
uv run scripts/smoke_test.py
```

The smoke test runs one seeded `PushCube-v1` episode with the CPU simulator, checks reset determinism and the 50-step time limit, then writes a random-policy video to `videos/smoke-test/`.

On macOS, rendering requires a Vulkan implementation. Install the Homebrew packages below if MoltenVK is not already available; the smoke test discovers the standard Apple Silicon and Intel Homebrew paths automatically.

```bash
brew install molten-vk vulkan-loader vulkan-tools
```

## Research question

> How do reward shaping and initial-state randomization affect PPO's learning speed, final performance, and generalization on robotic object pushing?

## Initial scope

- Environment: `PushCube-v1`
- Robot: Panda
- Observations: ground-truth `state`
- Controller: `pd_ee_delta_pose`, validated locally; the official baseline configuration still needs to be confirmed
- Algorithm: PPO
- Primary reward: ManiSkill dense reward
- Episode horizon: 50 steps

The first target is to reproduce a reliable ManiSkill PPO baseline before implementing or modifying the algorithm.

## Planned experiments

| ID | Setup | Reward | Initial-state randomization |
| --- | --- | --- | --- |
| A | Easy baseline | ManiSkill dense | Reduced |
| B | Standard task | ManiSkill dense | Standard |
| C | Reward ablation | Sparse or modified dense | Standard |

Every comparison will use the same interaction budget, evaluation protocol, and seeds. See [the experiment plan](docs/EXPERIMENT_PLAN.md) for the metrics and exit criteria.

## Roadmap

1. Validate the environment, controller, observations, metrics, and video recording.
2. Reproduce the official ManiSkill PPO baseline.
3. Implement a minimal PPO with an actor-critic MLP, GAE, clipped objective, value loss, and entropy bonus.
4. Compare dense and sparse/modified rewards over at least three seeds.
5. Compare direct training with an adaptive curriculum.
6. Consider `PushT-v1` only after the PushCube study is stable.

## Success criteria

The initial baseline phase is complete when it:

- reaches at least 90% deterministic evaluation success;
- can be reproduced from a clean environment with a fixed configuration and seed;
- reports success rate, return, episode length, and final cube-to-goal distance;
- produces random, early-training, and trained-policy videos;
- stores enough metadata to identify the code, environment, seed, and configuration used.

## Reproducibility principles

- Pin the Python and ManiSkill versions before the first benchmark.
- Keep configuration separate from training logic.
- Treat evaluation seeds separately from training seeds.
- Compare methods with the same environment-step budget.
- Keep generated runs, checkpoints, videos, and local tracking data out of Git.
- Record aggregate results across at least three seeds rather than selecting the best run.

## Hardware

Local macOS development is intended for CPU smoke tests and short validation runs. Parallel GPU training will target Linux with an NVIDIA RTX 3090. GPU simulation is not expected to work on macOS.

## References

- [ManiSkill documentation](https://maniskill.readthedocs.io/en/latest/)
- [ManiSkill `PushCube-v1` source](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/push_cube.py)
- [ManiSkill PPO baselines](https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/ppo)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
