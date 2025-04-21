#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --account=blanca-curc-gpu
#SBATCH --out=logs/polygloss.%j.out
#SBATCH --error=logs/polygloss.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.ginn@colorado.edu

module purge
module load miniforge
mamba activate polygloss
cd "/projects/$USER/polygloss"

python run.py "$1"

# torchrun --nproc_per_node=4 run.py --config ../configs/pretrain_base.cfg
