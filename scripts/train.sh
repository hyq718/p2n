#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
method="${1:?Usage: bash scripts/train.sh vanilla|p2n [Megatron arguments]}"
shift
case "$method" in vanilla|p2n) ;; *) echo "Unknown method: $method" >&2; exit 2 ;; esac
: "${TRAIN_DATA:?Set TRAIN_DATA to a Megatron indexed dataset prefix without .bin/.idx}"
: "${VALID_DATA:?Set VALID_DATA to a Megatron indexed validation prefix}"
: "${EOD_ID:?Set EOD_ID to the document-end token ID}"
size="${MODEL_SIZE:-70m}"
case "$size" in
  70m)
    architecture=(--num-layers 6 --hidden-size 512 --ffn-hidden-size 2048
                  --num-attention-heads 8 --num-query-groups 2 --untie-embeddings-and-output-weights)
    schedule=(--micro-batch-size 4 --global-batch-size 256 --train-iters 2836 --lr-warmup-iters 142)
    ;;
  150m)
    # Paper Appendix A, Table 6: 149,541,120 parameters with tied embeddings.
    # ceil(20 * parameters / (256 * 2048)) = 5705 steps.
    architecture=(--num-layers 12 --hidden-size 768 --ffn-hidden-size 3328
                  --num-attention-heads 12 --num-query-groups 4)
    schedule=(--micro-batch-size 8 --global-batch-size 256 --train-iters 5705 --lr-warmup-iters 286)
    ;;
  *) echo "Unknown MODEL_SIZE: $size (choose 70m or 150m)" >&2; exit 2 ;;
esac
output="${OUTPUT_ROOT:-checkpoints}/$size-$method"
args=("${architecture[@]}" "${schedule[@]}"
      --group-query-attention --kv-channels 64
      --normalization RMSNorm --norm-epsilon 1e-6 --qk-layernorm --swiglu --disable-bias-linear
      --position-embedding-type rope --rotary-base 1000000
      --tokenizer-type NullTokenizer --vocab-size 50303 --make-vocab-size-divisible-by 128 --eod-id "$EOD_ID"
      --seq-length 2048 --max-position-embeddings 2048
      --lr 0.0015 --min-lr 0.00015 --lr-decay-style cosine
      --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 --clip-grad 1.0
      --init-method-std 0.02 --hidden-dropout 0 --attention-dropout 0 --seed 42
      --bf16 --transformer-impl transformer_engine --attention-backend flash
      --cross-entropy-loss-fusion --no-masked-softmax-fusion
      --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
      --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather
      --train-data-path "$TRAIN_DATA" --valid-data-path "$VALID_DATA" --test-data-path "$VALID_DATA"
      --data-cache-path "${DATA_CACHE:-$output/data-cache}" --num-workers 4
      --no-create-attention-mask-in-dataloader
      --log-interval 1 --log-throughput --log-timers-to-tensorboard --tensorboard-log-interval 1
      --eval-interval 100 --eval-iters 4 --save-interval 200 --ckpt-format torch_dist
      --save "$output" --tensorboard-dir "$output/tensorboard"
      --wandb-project p2n --wandb-exp-name "${RUN_NAME:-native-$size-tpp20-$method}"
      --wandb-save-dir "$output/wandb")
if [[ "$method" == p2n ]]; then args+=(--p2n); fi
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then args+=(--load "$RESUME_CHECKPOINT"); fi
exec "${P2N_PYTHON:-python}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${GPUS:-8}" pretrain.py "${args[@]}" "$@"
