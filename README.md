# P2N

**Reusing deep representations for greater effective depth.**

## Overview

P2N (Previous to Next) feeds the previous token's deep core output into the
current token's prefix representation. The Transformer is split into prefix,
core, and suffix layers; Vanilla and P2N use the same physical layers and
parameters. This repository provides Qwen3-style pretraining on the official
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) stack, with Transformer
Engine and FlashAttention.

![P2N overview](assets/p2n-overview.png)

**Autoregressive decoding (right).** Let $x_t$ be the current token's prefix
representation and $h_{t-1}$ the previous token's core output. Since decoding
proceeds from left to right, $h_{t-1}$ is already available. P2N adds it to
$x_t$ before the core:

$$
h_t=\mathrm{Core}(x_t+h_{t-1}),\qquad h_0=0.
$$

The suffix and language-model head then predict the next token, and $h_t$ is
cached for the following decoding step. Each physical layer runs once per
current token; the feedback adds only one vector addition per step. Reusing
previously completed deep computation extends effective depth across tokens.

**Parallel training (left).** Applying the same recurrence directly during
teacher-forced training would require each token to wait for the preceding
token's core output. P2N preserves token parallelism with Jacobi iteration.
The prefix first processes the full sequence once to produce fixed inputs
$X$. A warm pass initializes the core states, followed by $K$ parallel updates:

$$
H^{(0)}=\mathrm{Core}(X),\qquad
H^{(k+1)}=\mathrm{Core}\!\left(X+\mathrm{ShiftPrev}(H^{(k)})\right),
\quad k=0,\ldots,K-1.
$$

Each update reads only the completed outputs of the preceding iteration, so
all token positions within that update can run in parallel. `ShiftPrev` shifts
these outputs one position right, filling the first position and document
boundaries with zeros. After the final update, the suffix runs once on
$H^{(K)}$ to compute the language-modeling loss. The core shares parameters
across the warm pass and all updates, with gradients through the complete
unrolled computation. Training samples $K$ uniformly from $\{2,3\}$;
validation uses $K=3$. Only the middle-third core is repeated, keeping the
extra training computation confined to that part of the model.

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

## Quick-start configuration (150M)

Set `MODEL_SIZE=150m` in the job example above. This configuration follows
Appendix A, Table 6 of the P2N paper and has **149,541,120 parameters**.
Vanilla and P2N use the same model and training settings.

### Model

| Setting | Value |
| --- | --- |
| Transformer layers | 12 |
| Hidden size | 768 |
| FFN size | 3,328 |
| Attention | GQA, 12 query heads / 4 KV heads, head dimension 64 |
| Vocabulary size | 50,304 |
| Input/output embeddings | Shared |
| Normalization | RMSNorm (`eps=1e-6`) and Q/K normalization |
| Activation | SwiGLU |
| Position encoding | RoPE, base `1e6` |
| Linear biases / dropout | Disabled / 0 |
| Initialization standard deviation | 0.02 |
| P2N core | Layers 5–8 (one-based, inclusive) |

### Training

| Setting | Value |
| --- | --- |
| Training budget | 2,991,063,040 tokens, approximately TPP20 |
| Optimizer steps | 5,705 |
| Sequence length | 2,048 |
| GPUs | 8, data parallelism |
| Global batch size | 256 sequences |
| Per-GPU micro-batch size | 8 sequences |
| Gradient accumulation | 4 micro-batches per optimizer step |
| Precision / attention backend | BF16 / Transformer Engine + FlashAttention |
| Optimizer | AdamW, betas `(0.9, 0.95)`, epsilon `1e-8` |
| Learning rate | `1.5e-3`, cosine decay to `1.5e-4` |
| Warmup | 286 steps, approximately 5% |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Activation recomputation | Disabled |
| Random seed | 42 |
