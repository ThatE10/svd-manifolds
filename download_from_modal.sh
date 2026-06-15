#!/usr/bin/env bash
# Download data from the Modal "fineweb-acts" volume.
#
# Usage:
#   ./download_from_modal.sh              # SVD results only (fast, ~GBs)
#   ./download_from_modal.sh --all        # everything incl. raw activations (~200 GB)
#   DATA_DIR=/path/to/dir ./download_from_modal.sh
#
# Prerequisites:
#   modal token new    # authenticate once

set -euo pipefail

VOLUME="fineweb-acts"
DATA_DIR="${DATA_DIR:-./data}"
ALL=false

for arg in "$@"; do
    [[ "$arg" == "--all" ]] && ALL=true
done

echo "Downloading to: $DATA_DIR"
mkdir -p "$DATA_DIR"

# Always download: metadata index + text-samples (needed by ActivationIndex)
echo "[1/3] metadata.jsonl ..."
modal volume get "$VOLUME" /metadata.jsonl "$DATA_DIR/metadata.jsonl"

echo "[2/3] text-samples/ ..."
modal volume get "$VOLUME" /text-samples "$DATA_DIR/text-samples"

# SVD results for all layers (much smaller than raw activations)
echo "[3/3] SVD results (layer 1, 12, 20, 24) ..."
for LAYER in 1 12 20 24; do
    REMOTE="/svd/layer_${LAYER}"
    LOCAL="$DATA_DIR/svd/layer_${LAYER}"
    if modal volume ls "$VOLUME" "$REMOTE" &>/dev/null; then
        echo "  layer $LAYER ..."
        mkdir -p "$LOCAL"
        modal volume get "$VOLUME" "$REMOTE" "$LOCAL"
    else
        echo "  layer $LAYER not found in volume, skipping"
    fi
done

if [[ "$ALL" == true ]]; then
    echo "[+] downloading raw activation shards (large - may take a while) ..."
    for LAYER in 1 12 20 24; do
        echo "  layer $LAYER activations ..."
        mkdir -p "$DATA_DIR/layer_${LAYER}"
        modal volume get "$VOLUME" "/layer_${LAYER}" "$DATA_DIR/layer_${LAYER}"
    done
fi

echo ""
echo "Done. Set OUTPUT_BASE=\"$(realpath "$DATA_DIR")\" in your notebooks."
