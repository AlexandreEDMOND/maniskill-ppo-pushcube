# PPO for ManiSkill PushCube

A reproducible study of how reward shaping and initial-state randomization affect PPO on ManiSkill's `PushCube-v1` task.

## Current status

The environment and official ManiSkill PPO pipeline have been validated locally. The official CUDA baseline for seed `9351` is complete and reaches 100% success on the fixed 20-episode deterministic evaluation. Its checkpoint and evaluation are kept as local experiment artifacts, outside Git.

The project now keeps that result as its reference run. The minimal custom PPO passes its CPU integration run: after 5,000 interactions, its mean return rises from 1.90 to 6.64 on five fixed seeds. A 50,000-step follow-up still reaches no task success on the shared 20-seed evaluation, so it is not yet a learning baseline. Additional official seeds will be run only when assembling final aggregate results.

See [ROADMAP.md](ROADMAP.md) for the authoritative progress tracker and next steps.

## Quick start

The project uses Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen
uv run scripts/smoke_test.py
```

The smoke test runs one seeded CPU-simulator episode, checks determinism and the 50-step limit, and writes a random-policy video to `videos/smoke-test/`.

On macOS, rendering requires Vulkan. If MoltenVK is not already available:

```bash
brew install molten-vk vulkan-loader vulkan-tools
```

## Scope

- Task: `PushCube-v1` with Panda
- Observations: ground-truth `state`
- Reference controller: `pd_joint_delta_pos`
- Reference reward: ManiSkill `normalized_dense`
- Algorithm: PPO
- Episode horizon: 50 steps

Full training runs require Linux, NVIDIA CUDA, and ManiSkill's `physx_cuda` backend; macOS is for smoke tests and short CPU validation runs.

## Key commands

```bash
# Validate the official pipeline locally; this is not a performance run.
uv run scripts/run_official_baseline.py --profile cpu-smoke
uv run scripts/evaluate.py runs/official-ppo-cpu-smoke/final_ckpt.pt

# Run one full official baseline on Linux/NVIDIA.
uv run scripts/run_official_baseline.py --profile official --seed 9351

# Train the minimal custom PPO locally, then evaluate it with the shared evaluator.
uv run scripts/train_custom_ppo.py
uv run scripts/evaluate.py runs/custom-ppo-cpu/final_ckpt.pt --no-video
```

## Documentation

- [Roadmap](ROADMAP.md): current status, milestones, and completion criteria
- [Custom PPO CPU configuration](configs/custom_ppo_cpu.json): short local training run
- [Experiment plan](docs/EXPERIMENT_PLAN.md): frozen comparison protocol and metrics
- [Official baseline](docs/OFFICIAL_BASELINE.md): upstream provenance and parameters
- [Toulouse GPU runbook](docs/TOULOUSE_GPU_RUNBOOK.md): Linux/RTX 3090 setup and commands

## References

- [ManiSkill documentation](https://maniskill.readthedocs.io/en/latest/)
- [ManiSkill PPO baselines](https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/ppo)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
