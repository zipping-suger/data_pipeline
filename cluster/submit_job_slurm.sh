#!/usr/bin/env bash

# submit_job_slurm.sh
# Usage: ./submit_job_slurm.sh

set -euo pipefail

echo "Generating SLURM job script..."

# Get a timestamp for unique log filename
timestamp=$(date +"%Y%m%d-%H%M%S")
logfile="slurm-${timestamp}.out"

# Create a dedicated tmp dir in scratch (avoids /tmp issues)
mkdir -p /cluster/scratch/yixili/tmp

cat <<EOT > job.sh
#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=96
#SBATCH --tmp=30G             # Request 30GB in /tmp (if needed)
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=4048
#SBATCH --output=$logfile
#SBATCH --mail-type=END
#SBATCH --mail-user=name@mail
#SBATCH --job-name="training-${timestamp}"

# Use a custom TMPDIR in scratch (safer than /tmp)
export TMPDIR=/cluster/scratch/yixili/tmp

# Clean up on exit (remove temporary files)
trap "rm -rf \$TMPDIR/*" EXIT

# Run the Singularity script
bash "/cluster/home/yixili/data_pipeline/cluster/run_singularity.sh"
EOT

echo "Submitting job to SLURM..."
job_output=$(sbatch job.sh)
job_id=$(echo "$job_output" | awk '{print $NF}')
echo "Submitted batch job $job_id"

# Remove the temporary job script
rm job.sh