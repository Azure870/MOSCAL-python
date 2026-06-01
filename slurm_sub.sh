#!/bin/bash
#SBATCH --job-name=l2
#SBATCH --partition=amd_m8_768
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --output=deom_%j.out
#SBATCH --error=deom_%j.err

source /public1/soft/modules/module.sh

# module load miniforge/24.11
source /public1/home/m8s001891/anaconda3/etc/profile.d/conda.sh

conda activate base

# python fermi-dt-filter_sub.py

python run.py
