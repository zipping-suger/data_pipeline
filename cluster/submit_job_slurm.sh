#!/usr/bin/env bash

# submit_job_slurm.sh
# Usage: ./submit_job_slurm.sh

set -euo pipefail

echo "Generating SLURM job script..."

# Get a timestamp for unique log filename
timestamp=$(date +"%Y%m%d-%H%M%S")
logfile="slurm-${timestamp}.out"

cat <<EOT > job.sh
#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=96
#SBATCH --tmp=12G  # Request 12GB in /tmp
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=4048
#SBATCH --output=$logfile
#SBATCH --mail-type=END
#SBATCH --mail-user=name@mail
#SBATCH --job-name="training-${timestamp}"

# Run the hardcoded run_singularity.sh script
bash "/cluster/home/yixili/data_pipeline/cluster/run_singularity.sh"
EOT

echo "Submitting job to SLURM..."
job_output=$(sbatch job.sh)
job_id=$(echo "$job_output" | awk '{print $NF}')
echo "Submitted batch job $job_id"

rm job.sh
