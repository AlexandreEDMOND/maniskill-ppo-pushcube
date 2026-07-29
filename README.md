# PPO for ManiSkill PushCube

A reproducible study of how reward shaping and initial-state randomization affect PPO on ManiSkill's `PushCube-v1` task.

## Current status

The environment and official ManiSkill PPO pipeline have been validated locally. The official CUDA baseline for seed `9351` is complete and reaches 100% success on the fixed 20-episode deterministic evaluation. Its checkpoint and evaluation are kept as local experiment artifacts, outside Git.

The project now keeps that result as its reference run. The minimal custom PPO uses bounded Gaussian actions and passes its CPU integration run. The CUDA custom trainer now uses periodic fixed-seed checkpoint selection, learning-rate decay, and conservative PPO updates; a 3-million-step pilot reaches 85% success on the shared evaluator. This is encouraging, but it is one training seed rather than a final comparison. Additional official seeds will be run only when assembling final aggregate results.

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

## 3M reward-alignment pilot

We compared the stabilized dense-reward control with a transparent time cost of
`-0.01` per environment step. Both selected checkpoints reach **85% success** on
the same 20 deterministic evaluation seeds. The time-cost condition is more stable
at the end of its run (85% versus 65%), but it does **not** yet make successful
episodes shorter: its selected policy needs 30.7 steps on average, compared with
15.6 for the control. It should therefore be treated as a promising stability
ablation—not as a confirmed improvement in efficiency—until it is repeated across
three training seeds.

![Checkpoint curves for the 3M pilot](docs/images/custom_ppo_time_penalty_3m.png)

The reusable plotting command is:

```bash
uv run scripts/plot_checkpoints.py \
  runs/custom-ppo-cuda-stabilized-3M-control/checkpoint_metrics.json \
  runs/custom-ppo-cuda-time-penalty-3M/checkpoint_metrics.json \
  --labels "Standard reward" "Time penalty (-0.01/step)" \
  --output docs/images/custom_ppo_time_penalty_3m.png
```

The shared evaluator also records representative MP4 videos locally under
`artifacts/evaluations/<run>/videos/` (experiment artifacts are deliberately not
committed). See [the pilot report](docs/RESULTS.md) for the exact measurements and
interpretation.

## Documentation

- [Roadmap](ROADMAP.md): current status, milestones, and completion criteria
- [Custom PPO CPU configuration](configs/custom_ppo_cpu.json): short local training run
- [Custom PPO CUDA configuration](configs/custom_ppo_cuda.json): vectorized matched-budget run
- [Time-cost reward configuration](configs/custom_ppo_cuda_time_penalty.json): 3M reward-alignment ablation
- [Pilot results](docs/RESULTS.md): 3M control versus time-cost comparison
- [Experiment plan](docs/EXPERIMENT_PLAN.md): frozen comparison protocol and metrics
- [Official baseline](docs/OFFICIAL_BASELINE.md): upstream provenance and parameters
- [Toulouse GPU runbook](docs/TOULOUSE_GPU_RUNBOOK.md): Linux/RTX 3090 setup and commands

## References

- [ManiSkill documentation](https://maniskill.readthedocs.io/en/latest/)
- [ManiSkill PPO baselines](https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/ppo)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
