#!/usr/bin/env bash

# run_singularity.sh
# Usage: ./run_singularity.sh <container_image> <pipeline_dir> <raw_dir> -- <script_args>
#
# Arguments:
#   container_image: Path to your Singularity image (e.g., /cluster/scratch/yixili/learn_to_reach)
#   pipeline_dir:    Host path to data_pipeline (e.g., /cluster/home/yixili/data_pipeline)
#   raw_dir:         Host path to raw_data       (e.g., /cluster/home/yixili/raw_data)
#   script_args:     Arguments passed to curobo_gen.py (e.g., cubby --neutral full-pipeline /raw_data/...)

set -euo pipefail

# Parse arguments
if [[ $# -lt 4 ]]; then
  echo "Error: Missing arguments."
  echo "Usage: $0 <container_image> <pipeline_dir> <raw_dir> -- <script_args>"
  exit 1
fi

CONTAINER_IMAGE="$1"
PIPELINE_DIR="$2"
RAW_DIR="$3"

# Determine position for script args
if [[ "$4" != "--" ]]; then
  echo "Error: Expected '--' before script arguments."
  exit 1
fi

# Shift past the first 4 args and collect remaining ones as script args
shift 4
SCRIPT_ARGS=("$@")

echo "Starting Singularity container: $CONTAINER_IMAGE"
singularity exec \
  --nv \
  --containall --writable-tmpfs \
  --bind "${PIPELINE_DIR}:/data_pipeline" \
  --bind "${RAW_DIR}:/raw_data" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONPATH="/data:\${PYTHONPATH:-}" \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env ACCEPT_EULA=Y \
  "${CONTAINER_IMAGE}" \
  python3 -u /data_pipeline/curobo_gen.py "${SCRIPT_ARGS[@]}"

echo "Completed run."

