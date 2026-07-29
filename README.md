# PPO for ManiSkill PushCube

A reproducible study of how reward shaping and initial-state randomization affect PPO on ManiSkill's `PushCube-v1` task.

## Current status

The environment and official ManiSkill PPO pipeline have been validated locally. The official CUDA baseline for seed `9351` is complete and reaches 100% success on the fixed 20-episode deterministic evaluation. Its checkpoint and evaluation are kept as local experiment artifacts, outside Git.

The project now keeps that result as its reference run. The minimal custom PPO uses bounded Gaussian actions and passes its CPU integration run. Its 50,000-step follow-up improves mean return, but still reaches no task success on the shared 20-seed evaluation. This is an integration check, not a performance comparison: 50,000 interactions are only 0.1% of the official 50-million-step budget. A custom learning baseline therefore requires a matched interaction budget on a vectorized CUDA trainer. Additional official seeds will be run only when assembling final aggregate results.

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

# Run the vectorized CUDA custom PPO on Linux/NVIDIA (long training run).
uv run scripts/train_custom_ppo.py \
  --config configs/custom_ppo_cuda.json \
  --output-dir runs/custom-ppo-cuda-9351
uv run scripts/evaluate.py runs/custom-ppo-cuda-9351/final_ckpt.pt

# First CUDA learning check (~1 minute on the validated RTX 3090 setup).
uv run scripts/train_custom_ppo.py \
  --config configs/custom_ppo_cuda.json \
  --total-timesteps 500000 \
  --output-dir runs/custom-ppo-cuda-500k
```

The CUDA configuration saves and evaluates a checkpoint every 500,000 requested
interactions on the fixed 20 CPU evaluation seeds. It keeps the selected model at
`runs/<run>/checkpoints/best_ckpt.pt`; `checkpoint_metrics.json` records success,
return, final distance, KL, clipping, losses, entropy, and action standard deviation.

## Documentation

- [Roadmap](ROADMAP.md): current status, milestones, and completion criteria
- [Custom PPO CPU configuration](configs/custom_ppo_cpu.json): short local training run
- [Custom PPO CUDA configuration](configs/custom_ppo_cuda.json): vectorized matched-budget run
- [Experiment plan](docs/EXPERIMENT_PLAN.md): frozen comparison protocol and metrics
- [Official baseline](docs/OFFICIAL_BASELINE.md): upstream provenance and parameters
- [Toulouse GPU runbook](docs/TOULOUSE_GPU_RUNBOOK.md): Linux/RTX 3090 setup and commands

## References

- [ManiSkill documentation](https://maniskill.readthedocs.io/en/latest/)
- [ManiSkill PPO baselines](https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/ppo)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
