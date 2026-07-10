# Official PPO baseline

This project treats the ManiSkill v3.0.1 release as the upstream reference. The source revision is pinned to commit [`a4a4f927`](https://github.com/haosulab/ManiSkill/tree/a4a4f9272ad64b1564035874b605ceb687b63ed8), and downloaded scripts are accepted only when their SHA-256 checksum matches the configuration.

## Controller decision

The official state-based `PushCube-v1` run in `examples/baselines/ppo/baselines.sh` invokes `ppo_fast.py` without a controller override. At the pinned revision, `ppo_fast.py` defaults to `pd_joint_delta_pos`. Therefore:

- the official reproduction uses `pd_joint_delta_pos` with an 8-dimensional action space;
- `pd_ee_delta_pose`, whose 7-dimensional action space passed the local smoke test, is reserved for a later controlled comparison;
- changing the controller is not allowed in the initial baseline result.

## Frozen official configuration

The machine-readable source of truth is [`configs/official_baseline.json`](../configs/official_baseline.json).

| Field | Value |
| --- | --- |
| Environment | `PushCube-v1` |
| Robot | Panda |
| Observation | `state`, shape `(1, 35)` for one environment |
| Controller | `pd_joint_delta_pos`, action shape `(8,)` |
| Reward | `normalized_dense` |
| Backend | `physx_cuda` |
| Upstream trainer | `ppo_fast.py` at `a4a4f927` |
| Seeds | `9351`, `4796`, `1788` |
| Budget | 50,000,000 environment interactions per seed |
| Parallel environments | 4096 |
| Rollout length | 4 steps per environment |
| PPO updates | 8 epochs, 32 minibatches |
| Evaluation | 16 environments, 50 steps |
| Evaluation frequency | 25 iterations = 409,600 environment interactions |
| Optimizations | CUDA graphs enabled |

The runner preserves upstream Weights & Biases tracking because the published command includes `--track`.

## Running the official reproduction

The following commands require Linux, an NVIDIA GPU, working CUDA/Vulkan drivers, and an authenticated Weights & Biases session:

```bash
uv sync --frozen

for seed in 9351 4796 1788; do
  uv run scripts/run_official_baseline.py --profile official --seed "$seed"
done
```

The runner refuses to start the official profile when CUDA is unavailable. This prevents a CPU run from being mislabeled as the official reproduction.

## Local CPU validation

The Mac validation uses the uncompiled upstream `ppo.py` from the same commit. The runner verifies its checksum and creates a transient copy whose only source change is `physx_cuda` → `physx_cpu`.

```bash
uv run scripts/run_official_baseline.py --profile cpu-smoke
uv run scripts/evaluate.py runs/official-ppo-cpu-smoke/final_ckpt.pt \
  --output-dir artifacts/evaluations/official-ppo-cpu-smoke
```

This profile deliberately uses only 1000 interactions, one environment, two update epochs, and one minibatch. It validates upstream training, checkpoint serialization, deterministic evaluation, metrics, and video recording. It is not a performance result.

The validated local run produced:

- 20 evaluation episodes on seeds `10000`–`10019`;
- 0% success, as expected after only 1000 interactions;
- mean return `1.8742 ± 0.5563`;
- mean final planar cube-to-goal distance `0.2000004 m`;
- 50 steps per episode;
- one deterministic 512×512 H.264 video.

## Evaluation protocol

[`scripts/evaluate.py`](../scripts/evaluate.py) loads checkpoints produced by both upstream `ppo.py` and `ppo_fast.py`. Actions are the deterministic actor mean. The evaluator records:

- success rate: fraction of episodes that satisfy success at least once, matching ManiSkill's `success_once` meaning;
- final success rate;
- undiscounted normalized-dense return;
- episode length;
- steps to first success for successful episodes;
- final planar Euclidean cube-to-goal distance;
- checkpoint SHA-256 and per-seed results;
- a deterministic video for seed `10000`.

This shared post-training evaluation uses the `physx_cpu` backend explicitly recorded in the configuration. It is separate from the upstream in-training evaluation, which remains on `physx_cuda`. Reports and videos are written under `artifacts/` and remain outside Git.

## Completion status

The baseline integration is reproduced locally end to end. The scientific reproduction remains incomplete until all three 50-million-interaction CUDA runs finish on the RTX 3090 and their checkpoints pass the fixed evaluator.
