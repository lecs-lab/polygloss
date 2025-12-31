#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --account=blanca-curc-gpu
#SBATCH --out=logs/%j.log
#SBATCH --error=logs/%j.log

module purge
module load miniforge 
mamba activate polygloss2
cd "/projects/$USER/polygloss"


python run.py experiments/active_learning/train_active.cfg
