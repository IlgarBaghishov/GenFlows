#!/bin/bash
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH -p gh
#SBATCH -t 36:00:00
#SBATCH -o ll_out
#SBATCH -A CHE21006

source /work/11316/rustamzade17/vista/miniforge3/etc/profile.d/conda.sh
conda activate genflows

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=29500

cd /work/11316/rustamzade17/vista/codes/GenFlows/examples/reservoirs/inpainting

export RESERVOIR_DATA_DIR=/scratch/11316/rustamzade17/SiliciclasticReservoirs

date
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=1 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  train.py
date
