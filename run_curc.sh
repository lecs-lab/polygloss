#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --account=blanca-curc-gpu
#SBATCH --out=logs/polygloss.%j.out
#SBATCH --error=logs/polygloss.%j.err

module purge
module load miniforge
mamba activate polygloss
cd "/projects/$USER/polygloss"

python run.py "$1"

# torchrun --nproc_per_node=4 run.py --config ../configs/pretrain_base.cfg
