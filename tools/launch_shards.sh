#!/usr/bin/env bash
# Run one review stage as N single-threaded CPU processes, one per shard.
# Usage: tools/launch_shards.sh N <run_original_review.py arguments...>
# Example:
#   tools/launch_shards.sh 16 --stage primary --prepared-root /data/prepared \
#       --output-root /results/primary --execute --resume
# Each shard logs to <output-root>/logs/shardK.log; the script waits for all
# shards and exits non-zero if any shard failed.
set -euo pipefail
count="$1"; shift
output_root=""
previous=""
for argument in "$@"; do
  if [ "$previous" = "--output-root" ]; then output_root="$argument"; fi
  previous="$argument"
done
if [ -z "$output_root" ]; then echo "pass --output-root" >&2; exit 2; fi
mkdir -p "$output_root/logs"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
pids=()
for shard in $(seq 1 "$count"); do
  python tools/run_original_review.py "$@" --device cpu --threads 1 --shard "$shard/$count" \
    > "$output_root/logs/shard$shard.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
