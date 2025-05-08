#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=4048
#SBATCH --mail-type=END
#SBATCH --mail-user=name@mail
#SBATCH --job-name="curobo_run-\$(date +"%Y-%m-%dT%H:%M:%S")"

cd $SCRATCH

singularity exec \
  --nv \
  --containall \
  --writable-tmpfs \
  --bind /cluster/home/yixili/data_pipeline:/data_pipeline \
  --bind /cluster/home/yixili/raw_data:/raw_data \
  --env "PYTHONPATH=/data:$PYTHONPATH" \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env ACCEPT_EULA=Y \
  learn_to_reach \
  python3 /data_pipeline/curobo_gen.py cubby --neutral full-pipeline /raw_data/cubby/neutral/curobo
