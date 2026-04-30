#!/bin/bash
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH -p gh
#SBATCH -t 24:00:00
#SBATCH -o ll_out
#SBATCH -A CHE25011

# --- environment ----------------------------------------------------------
# Dedicated env for this project: torch 2.9.0+cu128, torchvision 0.24.0,
# accelerate, pyarrow, huggingface_hub, and genflows (-e .).
source /work/11316/rustamzade17/vista/miniforge3/etc/profile.d/conda.sh
conda activate genflows

# --- DDP rendezvous -------------------------------------------------------
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=29500

cd /work/11316/rustamzade17/vista/codes/GenFlows/examples/reservoirs/standard

# Dataset and per-split cond caches already on $SCRATCH:
#   /scratch/11316/rustamzade17/SiliciclasticReservoirs/{splits,_cond_cache,<layer_type>/...}
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
