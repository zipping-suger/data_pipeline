#!/usr/bin/env bash

# run_singularity.sh
# Hardcoded paths and arguments for simplicity

set -euo pipefail

# Hardcoded paths and arguments
CONTAINER_IMAGE="/cluster/scratch/yixili/learn_to_reach"
PIPELINE_DIR="/cluster/home/yixili/data_pipeline"
RAW_DIR="/cluster/home/yixili/raw_data"

echo "Starting Singularity container: $CONTAINER_IMAGE"
singularity exec \
  --nv \
  --containall --writable-tmpfs \
  --bind "${PIPELINE_DIR}:/data_pipeline" \
  --bind "${RAW_DIR}:/raw_data" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONPATH="/data_pipeline:\${PYTHONPATH:-}" \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env ACCEPT_EULA=Y \
  "${CONTAINER_IMAGE}" \
  /usr/bin/python3 -u /data_pipeline/ompl_gen.py cubby full-pipeline /raw_data/cubby/single/

echo "Completed run."
