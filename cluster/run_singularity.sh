#!/usr/bin/env bash

# run_singularity.sh
# Hardcoded paths and arguments for singularity execution

set -euo pipefail

# --- Hardcoded Paths ---
CONTAINER_IMAGE="/cluster/scratch/yixili/learn_to_reach.sif"
PIPELINE_DIR="/cluster/home/yixili/data_pipeline"
RAW_DIR="/cluster/home/yixili/raw_data"
SCRATCH_TMP="/cluster/scratch/yixili/tmp"  # Custom tmp dir (avoid /tmp issues)

# --- Environment and Task Definitions ---
# ENVS=("cubby" "dresser" "tabletop")
ENVS=("dresser")

declare -A TASKS=(
  [task-oriented]=task
  # [neutral]=neutral
  # [free-space]=free
  # [mixed]=mixed
)

# --- Ensure Scratch TMP Exists ---
mkdir -p "$SCRATCH_TMP"

echo "Starting Singularity container: $CONTAINER_IMAGE"

for ENV in "${ENVS[@]}"; do
  # Update DATA_SAVE_DIR based on environment
  DATA_SAVE_DIR="/raw_data/${ENV}_tasks/"
  
  for TYPE in "${!TASKS[@]}"; do
    OUTDIR="${TASKS[$TYPE]}"
    OUTPUT_PATH="${DATA_SAVE_DIR}${OUTDIR}/"
    HOST_OUTPUT_DIR="${RAW_DIR}/${ENV}_tasks/${OUTDIR}"
    
    echo "=== Running $ENV environment - $TYPE task (Output: $OUTPUT_PATH) ==="
    
    # Create output directory first on host system
    echo "Creating output directory: $HOST_OUTPUT_DIR"
    mkdir -p "$HOST_OUTPUT_DIR"
    
    # --- Run Singularity with Proper TMPDIR ---
    # Bind scratch to /tmp in container
    singularity exec \
      --nv \
      --containall --writable-tmpfs \
      --bind "${PIPELINE_DIR}:/data_pipeline" \
      --bind "${RAW_DIR}:/raw_data" \
      --bind "${SCRATCH_TMP}:/tmp" \
      --env PYTHONUNBUFFERED=1 \
      --env PYTHONPATH="/data_pipeline:\${PYTHONPATH:-}" \
      --env NVIDIA_DRIVER_CAPABILITIES=all \
      --env ACCEPT_EULA=Y \
      --env TMPDIR="$SCRATCH_TMP" \
      "${CONTAINER_IMAGE}" \
      /usr/bin/python3 -u /data_pipeline/task_gen.py "$ENV" "$TYPE" full-pipeline "$OUTPUT_PATH"

    echo "=== Completed $ENV - $TYPE ==="
  done
done

echo "All runs completed."