#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --ntasks=3
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --account=blanca-curc-gpu
#SBATCH --out=train_glosslm.%j.out
#SBATCH --error=train_glosslm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.ginn@colorado.edu

module purge
module load anaconda
conda activate AutoIGT
cd "/projects/migi8081/glosslm/src"


torchrun --nproc_per_node=4 run.py --config ../configs/pretrain_base.cfg
