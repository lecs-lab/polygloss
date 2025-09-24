#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=4
#SBATCH --time=5-00:00:00
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --account=blanca-curc-gpu
#SBATCH --out=logs/%j.log
#SBATCH --error=logs/%j.log

module purge
module load miniforge
mamba activate polygloss
cd "/projects/$USER/polygloss"


for limit in 100 500 1000 1500 2000 
do
    echo "Running with training set of $limit"
    python run.py /projects/enri8153/polygloss/experiments/igt_unsegmented_no_pretrain/pretrain_monolingual.cfg -o limit=$limit
done
