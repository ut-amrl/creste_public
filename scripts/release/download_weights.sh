#!/bin/bash

# Exit on error
set -e

# Check argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <model_name>"
    echo "Example: $0 creste_rgbd"
    exit 1
fi

MODEL_NAME="$1"
DEST_DIR="pretrained_weights"
mkdir -p "$DEST_DIR"

# Define download URLs
case "$MODEL_NAME" in
  creste_rgbd)
    URL="https://web.corral.tacc.utexas.edu/texasrobotics/web_CREStE/pretrained_models/traversability_model_trace_distill128_cfs.pt"
    FILENAME="rgbd_distill128_cfs.pth"
    ;;
  *)
    echo "Unknown model name: $MODEL_NAME"
    exit 1
    ;;
esac

# Download file
echo "Downloading $MODEL_NAME weights..."
curl -L -o "$DEST_DIR/$FILENAME" "$URL"

# Print path
echo "Downloaded to: $(realpath "$DEST_DIR/$FILENAME")"
