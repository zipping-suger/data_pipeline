#!/usr/bin/env bash

# submit_job_slurm.sh
# Usage: ./submit_job_slurm.sh <container_image> <pipeline_dir> <raw_dir> -- <script_args>

set -euo pipefail

echo "Generating SLURM job script..."

# Get a timestamp for unique log filename
timestamp=$(date +"%Y%m%d-%H%M%S")
logfile="slurm-${timestamp}.out"

cat <<EOT > job.sh
#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=4048
#SBATCH --output=$logfile
#SBATCH --mail-type=END
#SBATCH --mail-user=name@mail
#SBATCH --job-name="training-${timestamp}"

# Pass the container image and arguments to run_singularity.sh
bash "\$1/cluster/run_singularity.sh" "\$1" "\$2" "\$3" -- "\${@:4}"
EOT

echo "Submitting job to SLURM..."
job_output=$(sbatch job.sh)
job_id=$(echo "$job_output" | awk '{print $NF}')
echo "Submitted batch job $job_id"

rm job.sh
echo "Waiting for job $job_id to complete..."

# Poll SLURM job status
while squeue -j "$job_id" 2>/dev/null | grep -q "$job_id"; do
  sleep 10
done

echo "Job completed. Output:"
cat "$logfile"

