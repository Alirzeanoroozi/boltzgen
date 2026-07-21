#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --time=72:00:00
#SBATCH --mem=40G
#SBATCH --output=logs/comp.out
#SBATCH --error=logs/comp.out

# Initialise environment and modules
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate bg_macrel
export LD_LIBRARY_PATH=${CONDA_BASE}/lib

nvidia-smi

# boltzgen run inputs/config_KPC3.yaml --output outputs/hallucination_run --protocol peptide-anything --num_designs 100 --budget 10

# boltzgen run inputs/config_KPC3.yaml --output outputs/test_design --protocol peptide-anything --num_designs 20 --budget 2 --steps design
python design.py
# python inverse_folding.py
# python folding.py
# python analysis.py
# python filtering.py