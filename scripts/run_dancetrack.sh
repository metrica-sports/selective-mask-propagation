#!/bin/bash
# Run SAM-Deep-EIoU on DanceTrack.
# Each step processes all clips in one invocation (models load once).
#
# Usage:
#   ./run_dancetrack.sh val sam3
#   ./run_dancetrack.sh val sam2 --tracker bytetrack
#   ./run_dancetrack.sh val sam3 --tracker sort
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 val|test sam2|sam3 [--tracker deepeiou|bytetrack|sort]"
    exit 1
fi

SPLIT="$1"
if [ "$SPLIT" != "val" ] && [ "$SPLIT" != "test" ]; then
    echo "Error: split must be val or test"
    exit 1
fi

if [ "$2" != "sam2" ] && [ "$2" != "sam3" ]; then
    echo "Error: SAM version must be sam2 or sam3"
    exit 1
fi

SAM_FLAG=""
[ "$2" = "sam3" ] && SAM_FLAG="--sam3"

TRACKER_FLAG=""
if [ "$3" = "--tracker" ] && [ -n "$4" ]; then
    TRACKER_FLAG="--tracker $4"
fi

cd "$(dirname "$0")/.."
INPUT="data/dancetrack/${SPLIT}/dancetrack*"

uv run python -m smp.dancetrack --input "$INPUT" $SAM_FLAG $TRACKER_FLAG --precomputed --skip-existing

echo "Done."
