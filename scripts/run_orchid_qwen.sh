#!/bin/bash

#SBATCH --job-name=polygloss
#SBATCH --output=./slurm-out/train_%j.out
#SBATCH --time=7-00:00:00
#SBATCH --partition=flame
#SBATCH --qos=flame-8gpu_qos
#SBATCH --account=gneubig
#SBATCH --gres=gpu:8
#SBATCH --mem=48GB
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mail-user=lindiat@andrew.cmu.edu
#SBATCH --mail-type=START,END,FAIL

model_dir="/project/flame/lindiat/polygloss/models"
config="experiments/qwen_g+s_interleaved/pretrain.cfg"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate polygloss
export HF_HOME="/project/flame/lindiat/huggingface"

export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

torchrun \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    run.py $config --overrides batch_size=24 model_dir=$model_dir
