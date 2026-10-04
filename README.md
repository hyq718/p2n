# P2N

**Reusing deep representations for greater effective depth.**

## Overview

P2N reuses a shared Transformer core through Jacobi updates while the prefix
and suffix run once. Vanilla and P2N use the same physical layers and parameters.
This repository implements Qwen3-style pretraining on the official
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) stack, with Transformer
Engine and FlashAttention.

![P2N overview](assets/p2n-overview.png)

For prefix output $X$, the core computation is:

$$
H^{(0)}=\mathrm{Core}(X),\qquad
H^{(k+1)}=\mathrm{Core}\!\left(X+\mathrm{ShiftPrev}(H^{(k)})\right).
$$

`ShiftPrev` shifts representations one token right, zeros the first position,
and resets after EOD. Training samples $K\in\{2,3\}$ per optimizer step;
validation uses $K=3$. Gradients flow through every pass. The figure also
illustrates inference; this release implements pretraining only.

## Submit training jobs

After [setup](#setup), run from the repository root. Replace the data, output,
partition, and EOD values with your own:

```bash
source .venv/bin/activate
export TRAIN_DATA=/path/to/your_pt_data/train
export VALID_DATA=/path/to/your_pt_data/valid
export EOD_ID=0  # replace with your tokenizer's EOD token ID
export OUTPUT_ROOT=/path/to/your_output/checkpoints
export DATA_CACHE=/path/to/your_output/data_cache
export MODEL_SIZE=150m
export WANDB_MODE=disabled

METHOD=vanilla sbatch --partition=your_gpu_partition --job-name=150m-vanilla scripts/train.sbatch
METHOD=p2n sbatch --partition=your_gpu_partition --job-name=150m-p2n scripts/train.sbatch
```

The Slurm template requests eight GPUs. Adjust its CPU, memory, and time
requests for your cluster. Without Slurm, use `GPUS=8 bash scripts/train.sh vanilla`
or `GPUS=8 bash scripts/train.sh p2n` in the same environment.
To log online, set `WANDB_MODE=online`, `WANDB_ENTITY=your_wandb_entity`,
and optionally `WANDB_RUN_GROUP=your_experiment`, then run `wandb login`.

Data is **not included**. Supply Megatron indexed datasets: each prefix above
must have matching `.bin` and `.idx` files; omit these extensions in the paths.
The supplied recipes use vocabulary size 50,304 and `NullTokenizer` for
pretokenized data. Use matching token IDs and EOD metadata. Both methods use
causal attention and continuous position IDs; P2N feedback resets after EOD.

## Setup

Use Linux, Python 3.11, a BF16-capable NVIDIA GPU, and a compatible CUDA toolkit
with cuDNN development headers. Start from an environment providing NVIDIA
Apex's CUDA extensions, including `fused_weight_gradient_mlp_cuda`.

```bash
git clone --recurse-submodules https://github.com/hyq718/p2n.git
cd p2n
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install torch==2.6.0 packaging ninja pybind11 wheel
pip install --no-build-isolation -r requirements.txt
```

Megatron-LM is pinned to `core_v0.13.0`
(`c550cf6c41c31cd3ec72e05c25ea0c979f2b6631`) as an unmodified Git submodule.
The tested runtime uses PyTorch 2.6.0, Transformer Engine 2.13.0,
FlashAttention 2.7.4.post1, and Apex CUDA extensions.

## Model configurations

| `MODEL_SIZE` | Parameters | Layers | Hidden / FFN | Q / KV heads | Embeddings | Microbatch | Steps / tokens |
| --- | ---: | ---: | --- | --- | --- | ---: | --- |
| `70m` (default) | 74,325,248 | 6 | 512 / 2,048 | 8 / 2 | Untied | 4 | 2,836 / 1.487B |
| `150m` | 149,541,120 | 12 | 768 / 3,328 | 12 / 4 | Tied | 8 | 5,705 / 2.991B |

The 150M architecture follows Appendix A, Table 6 of the P2N paper.
Both recipes target 20 tokens per parameter with sequence length 2,048,
global batch 256, seed 42, BF16, GQA, Q/K normalization, RMSNorm, SwiGLU,
head dimension 64, RoPE base `1e6`, and zero dropout. AdamW uses betas
`(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1`, and clipping at `1.0`.
Peak LR is `1.5e-3`, with 5% warmup and cosine decay to `1.5e-4`.
With eight GPUs, accumulation is eight microbatches for 70M and four for 150M.

The default P2N core is the middle third: layers 3–4 for 70M and 5–8 for 150M.
To select a different range, pass **one-based, inclusive** bounds:

```bash
MODEL_SIZE=150m bash scripts/train.sh p2n --p2n-core-start 5 --p2n-core-end 8
```

Extra arguments are passed to Megatron; apply the same overrides to both
methods for an aligned comparison. P2N supports dense pretraining with
`TP=PP=CP=1`, zero dropout, and no activation recomputation, FP8, CPU
offloading, THD packing, or inference cache. Vanilla retains upstream support.

## Checkpoints and validation

Checkpoints are saved every 200 steps under
`$OUTPUT_ROOT/$MODEL_SIZE-$METHOD`; validation runs every 100 steps.
Resume from a checkpoint directory:

```bash
MODEL_SIZE=150m RESUME_CHECKPOINT=/path/to/your_checkpoint/150m-p2n \
  bash scripts/train.sh p2n
```

For W&B resume, also set `WANDB_RUN_ID=your_run_id` and `WANDB_RESUME=allow`.
Compare losses at equal steps or tokens. Use measured seconds per step for
speed: Megatron's default TFLOPS estimate does not count P2N's repeated core.

```bash
pip install pytest
python -m pytest -q tests/test_recurrence.py
python tests/check_gpu.py  # one GPU; checks TE/FlashAttention outputs and gradients
```

The repository contains the pretraining adapter (`pretrain.py`), recurrence
(`p2n/`), launchers (`scripts/`), checks (`tests/`), and overview figure
(`assets/`). Megatron-LM stays in `vendor/Megatron-LM` as a submodule.
See [LICENSE](LICENSE) for licensing terms and preserve dependency notices.
