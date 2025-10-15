#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --account=blanca-curc-gpu
#SBATCH --out=logs/%j.log
#SBATCH --error=logs/%j.log

module purge
module load miniforge
mamba activate polygloss
cd "/projects/$USER/polygloss"


python continual_peft.py /projects/enri8153/polygloss/experiments/continual_peft_sequential/peft.cfg
