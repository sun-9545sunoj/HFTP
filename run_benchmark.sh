#!/usr/bin/env bash
# Leak-free architecture benchmark: train + evaluate each model on the
# sequential LOBSTER parquet. Results are appended to reports/results.jsonl.
#
# Usage: nohup ./run_benchmark.sh [K] [STOCKS] [MODELS...] > reports/benchmark.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"

K="${1:-20}"
STOCKS="${2:-AAPL,AMZN,GOOG}"
shift $(( $# < 2 ? $# : 2 ))
if [ $# -gt 0 ]; then MODELS=("$@"); else MODELS=(deeplob translob transformer); fi

SEED="${SEED:-}"   # optional: SEED=1 ./run_benchmark.sh ...
SEED_ARG=""; SEED_TAG=""
if [ -n "$SEED" ]; then SEED_ARG="--seed $SEED"; SEED_TAG="-s$SEED"; fi
PY=".venv/bin/python -u"
PARQUET="src/data/LOBSTER_sequential.parquet"
TAG="${STOCKS//,/}"

for MODEL in "${MODELS[@]}"; do
  echo "================ $MODEL | k=$K | $STOCKS | seed=${SEED:-none} | $(date) ================"
  $PY train.py --model_type "$MODEL" --dataset_type parquet --file_path "$PARQUET" \
    --k "$K" --stocks "$STOCKS" --pooling last \
    --batch_size 256 --epochs 15 --lr 1e-4 --weight_decay 0.01 --dropout 0.2 \
    --loss_type weighted_ce $SEED_ARG

  CKPT=$(ls -t checkpoints/enhanced-parquet-"$MODEL"-k"$K"-"$TAG""$SEED_TAG"-best-*.ckpt | head -1)
  echo "-> Evaluating $CKPT"
  $PY evaluate_sequential.py --checkpoint "$CKPT" --k "$K" --stocks "$STOCKS" --split test
done
echo "================ done $(date) ================"
