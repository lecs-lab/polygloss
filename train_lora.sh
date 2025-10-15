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
#SBATCH --out=toto_logs/%j.log
#SBATCH --error=toto_logs/%j.log

module purge
module load miniforge
mamba activate polygloss
cd "/projects/$USER/polygloss"


python run.py /projects/enri8153/polygloss/experiments/polygloss_peft/peft.cfg
