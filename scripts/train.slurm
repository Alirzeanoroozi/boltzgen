#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:tesla_a100:7
#SBATCH --time=72:00:00
#SBATCH --mem=300G
#SBATCH --output=logs/train_small.out
#SBATCH --error=logs/train_small.out

# Initialise environment and modules
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate bg_macrel
export LD_LIBRARY_PATH=${CONDA_BASE}/lib

srun python -u train.py
