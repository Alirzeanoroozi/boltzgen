#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:tesla_a100:2
#SBATCH --time=72:00:00
#SBATCH --mem=40G
#SBATCH --output=logs/comp.out
#SBATCH --error=logs/comp.out

# Initialise environment and modules
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate new_bg
export LD_LIBRARY_PATH=${CONDA_BASE}/lib

nvidia-smi

# boltzgen run example/peptide_hallucination/hallucination.yaml \
#  --output workbench/hallucination_run \
#  --protocol peptide-anything \
#  --num_designs 100 \
#  --budget 10 \
#  --steps design

# python src/boltzgen/resources/main.py workbench/hallucination_run/config/design.yaml

python generate.py 
