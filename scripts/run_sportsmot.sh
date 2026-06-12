#!/bin/bash
# Run SAM-Deep-EIoU on SportsMOT.
# Each step processes all clips in one invocation (models load once).
#
# Usage:
#   ./run_sportsmot.sh val sam3
#   ./run_sportsmot.sh train sam2
#   ./run_sportsmot.sh test sam3
#   ./run_sportsmot.sh test sam3 2        # partition 2 only (for parallel runs)
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 val|train|test sam2|sam3 [partition]"
    exit 1
fi

SPLIT="$1"
if [ "$SPLIT" != "val" ] && [ "$SPLIT" != "train" ] && [ "$SPLIT" != "test" ]; then
    echo "Error: split must be val, train, or test"
    exit 1
fi

if [ "$2" != "sam2" ] && [ "$2" != "sam3" ]; then
    echo "Error: SAM version must be sam2 or sam3"
    exit 1
fi

SAM_FLAG=""
[ "$2" = "sam3" ] && SAM_FLAG="--sam3"

cd "$(dirname "$0")/.."

# Handle partitioned test set runs
if [ -n "$3" ]; then
    PARTITION_FILE="partitions/${SPLIT}_${3}.txt"
    if [ ! -f "$PARTITION_FILE" ]; then
        echo "Partition file not found: $PARTITION_FILE"
        echo "Run: uv run python scripts/partition_clips.py data/sportsmot/dataset/$SPLIT <N>"
        exit 1
    fi
    mapfile -t CLIPS < "$PARTITION_FILE"
    echo "Partition $3: ${#CLIPS[@]} clips"
    INPUT_ARGS=("${CLIPS[@]}")
else
    INPUT_ARGS=("data/sportsmot/dataset/${SPLIT}/*")
fi

uv run python -m sam_deep_eiou.sportsmot --input "${INPUT_ARGS[@]}" --step detect --precomputed --gta $SAM_FLAG
uv run python -m sam_deep_eiou.sportsmot --input "${INPUT_ARGS[@]}" --step track --precomputed --gta $SAM_FLAG
uv run python -m sam_deep_eiou.sportsmot --input "${INPUT_ARGS[@]}" --step sam --precomputed --gta $SAM_FLAG
uv run python -m sam_deep_eiou.sportsmot --input "${INPUT_ARGS[@]}" --step merge --precomputed --gta $SAM_FLAG
uv run python -m sam_deep_eiou.sportsmot --input "${INPUT_ARGS[@]}" --step pose --precomputed --gta $SAM_FLAG
uv run python -m sam_deep_eiou.sportsmot --input "${INPUT_ARGS[@]}" --step jersey --precomputed --gta $SAM_FLAG
uv run python -m sam_deep_eiou.sportsmot --input "${INPUT_ARGS[@]}" --step classify --precomputed --gta $SAM_FLAG
uv run python -m sam_deep_eiou.sportsmot --input "${INPUT_ARGS[@]}" --step embed --precomputed --gta $SAM_FLAG
uv run python -m sam_deep_eiou.sportsmot --input "${INPUT_ARGS[@]}" --step match --precomputed --gta $SAM_FLAG
uv run python -m sam_deep_eiou.sportsmot --input "${INPUT_ARGS[@]}" --step interp --precomputed --gta $SAM_FLAG

if [ "$SPLIT" != "test" ]; then
    uv run python -m sam_deep_eiou.sportsmot --input "${INPUT_ARGS[@]}" --step eval --precomputed --gta $SAM_FLAG
fi

echo "Done."
if [ "$SPLIT" = "test" ]; then
    echo "Run build_submission.py to produce submission zip."
else
    echo "Run aggregate_eval.py to see results."
fi
