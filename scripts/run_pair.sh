#!/usr/bin/env bash
# Usage: bash scripts/run_pair.sh DATA_ROOT RUN_NAME PER_CATEGORY REPLICAS BATCH_SIZE MAX_NEW_TOKENS
set -euo pipefail
ROOT="${1:?data root required}"
RUN="${2:?run name required}"
PER_CATEGORY="${3:-0}"
REPLICAS="${4:-4}"
BATCH_SIZE="${5:-2}"
MAX_NEW_TOKENS="${6:-256}"
SCRIPTS="$(cd -- "$(dirname -- "$0")" && pwd)"
PYTHON="$ROOT/envs/inference/bin/python"
export HF_HOME="$ROOT/hf-cache" HF_DATASETS_CACHE="$ROOT/hf-cache/datasets" TMPDIR="$ROOT/tmp"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
[[ -f "$ROOT/logs/assets-ready" ]] || { echo 'Assets are not ready'; exit 1; }
mkdir -p "$ROOT/runs/$RUN"
# This launcher refuses occupied GPUs rather than mixing with existing jobs.
for (( gpu=0; gpu<2*REPLICAS; gpu++ )); do
  memory="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
  if (( memory > 256 )); then echo "GPU $gpu occupied: $memory MiB"; exit 1; fi
done
pids=()
for model in base safe; do
  offset=0
  [[ "$model" == safe ]] && offset="$REPLICAS"
  for (( shard=0; shard<REPLICAS; shard++ )); do
    CUDA_VISIBLE_DEVICES="$((offset+shard))" "$PYTHON" "$SCRIPTS/infer_llava_pair.py" \
      --root "$ROOT" --run "$RUN" --model "$model" --shard "$shard" --num-shards "$REPLICAS" \
      --per-category "$PER_CATEGORY" --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" \
      > "$ROOT/runs/$RUN/$model-$shard.log" 2>&1 &
    pids+=("$!")
  done
done
printf '%s\n' "${pids[@]}" > "$ROOT/runs/$RUN/worker-pids.txt"
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then echo 'Some workers failed. Inspect per-worker logs; incomplete results not summarized.'; exit 1; fi
expected=590
(( PER_CATEGORY > 0 )) && expected=$((20*PER_CATEGORY))
"$PYTHON" "$SCRIPTS/summarize_pair.py" --run-dir "$ROOT/runs/$RUN" --expected "$expected" \
  > "$ROOT/runs/$RUN/summary.log"
touch "$ROOT/runs/$RUN/complete"
echo "Completed $RUN: $ROOT/runs/$RUN/summary.json"
