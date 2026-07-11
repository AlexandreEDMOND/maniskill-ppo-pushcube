# Reproducing PPO for Robotic Object Pushing in ManiSkill

This project studies how reward shaping and initial-state randomization affect PPO on ManiSkill's `PushCube-v1` task. The end goal is a small, reproducible deep reinforcement learning study rather than a one-off training run.

> **Status:** the official baseline pipeline is validated locally; the full three-seed CUDA reproduction is pending. No custom PPO implementation has been added yet.

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
- Official baseline controller: `pd_joint_delta_pos`
- Later controller comparison: `pd_ee_delta_pose`, already validated locally
- Algorithm: PPO
- Primary reward: ManiSkill normalized dense reward
- Episode horizon: 50 steps

The first target is to reproduce a reliable ManiSkill PPO baseline before implementing or modifying the algorithm.

## Official baseline

The reference is pinned to ManiSkill v3.0.1 commit `a4a4f927`. Its published `PushCube-v1` state baseline uses `ppo_fast.py`, `pd_joint_delta_pos`, 4096 CUDA environments, 50 million interactions, and seeds `9351`, `4796`, and `1788`.

```bash
# End-to-end local validation; not a performance run
uv run scripts/run_official_baseline.py --profile cpu-smoke
uv run scripts/evaluate.py runs/official-ppo-cpu-smoke/final_ckpt.pt

# One exact official run on Linux/NVIDIA
uv run scripts/run_official_baseline.py --profile official --seed 9351
```

See [the official baseline protocol](docs/OFFICIAL_BASELINE.md) for provenance, all frozen parameters, metric definitions, and the precise distinction between the CPU validation and the CUDA reproduction.

For the exact commands to run on the Toulouse RTX 3090 machine, see [the Toulouse GPU runbook](docs/TOULOUSE_GPU_RUNBOOK.md).

## Planned experiments

| ID | Setup | Reward | Initial-state randomization |
| --- | --- | --- | --- |
| A | Easy baseline | ManiSkill dense | Reduced |
| B | Standard task | ManiSkill dense | Standard |
| C | Reward ablation | Sparse or modified dense | Standard |

Every comparison will use the same interaction budget, evaluation protocol, and seeds. See [the experiment plan](docs/EXPERIMENT_PLAN.md) for the metrics and exit criteria.

## Roadmap

1. Validate the environment, controller, observations, metrics, and video recording. **Done.**
2. Validate the official ManiSkill PPO pipeline locally. **Done.**
3. Run the official 50-million-step baseline for all three seeds on Linux/NVIDIA.
4. Implement a minimal PPO with an actor-critic MLP, GAE, clipped objective, value loss, and entropy bonus.
5. Compare dense and sparse/modified rewards over at least three seeds.
6. Compare direct training with an adaptive curriculum.
7. Consider `PushT-v1` only after the PushCube study is stable.

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
