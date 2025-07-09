#!/usr/bin/env bash

# run_singularity.sh
# Hardcoded paths and arguments for simplicity

set -euo pipefail

# Hardcoded paths and arguments
CONTAINER_IMAGE="/cluster/scratch/yixili/learn_to_reach"
PIPELINE_DIR="/cluster/home/yixili/data_pipeline"
RAW_DIR="/cluster/home/yixili/raw_data"

# For task_gen include free, mixed, task-oriented, free-space
# For ompl_gen, include task-oriented, neutral to minimize the problem of multi modality

declare -A TASKS=(
  [free-space]=free
  [mixed]=mixed
  [task-oriented]=task
  [neutral]=neutral
)

echo "Starting Singularity container: $CONTAINER_IMAGE"
for TYPE in "${!TASKS[@]}"; do
  OUTDIR="${TASKS[$TYPE]}"
  echo "Running $TYPE..."
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
    /usr/bin/python3 -u /data_pipeline/task_gen.py cubby "$TYPE" full-pipeline "/raw_data/single_cubby_tasks2/$OUTDIR/"
  echo "Completed $TYPE."
done

echo "All runs completed."
