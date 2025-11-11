#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --account=blanca-curc-gpu
#SBATCH --out=logs/%j.log
#SBATCH --error=logs/%j.log

module purge
module load miniforge
mamba activate polygloss
cd "/projects/$USER/polygloss"

set -x

export MODELS_DIR=/scratch/alpine/migi8081/polygloss/
export GLOTTOCODE=$2

wandb agent $1
