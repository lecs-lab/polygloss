#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100_3g.40gb
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=4000m
#SBATCH --time=7-00:00:00
#SBATCH --qos=blanca-blast-lecs
#SBATCH --partition=blanca-blast-lecs
#SBATCH --account=blanca-blast-lecs
#SBATCH --out=logs/polygloss.%j.out
#SBATCH --error=logs/polygloss.%j.err

module purge
module load miniforge
mamba activate polygloss
cd "/projects/$USER/polygloss"

python run.py "$1"

# torchrun --nproc_per_node=4 run.py --config ../configs/pretrain_base.cfg
