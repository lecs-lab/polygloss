#!/bin/bash
#SBATCH --job-name=polygloss
#SBATCH --output=./slurm-out/train_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100_80GB:4
#SBATCH --mem=48GB
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=lindiat@andrew.cmu.edu
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=general
#SBATCH --exclude=babel-q5-16,babel-q5-20


conda init bash
source ~/.bashrc
conda activate polygloss
export HF_HOME="/data/tir/projects/tir1/users/lindiat/huggingface"

export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export OMP_NUM_THREADS=6
export PYTHONUNBUFFERED=1

config="experiments/byt5_g+s_multitask/pretrain.cfg"

torchrun \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    run.py $config --overrides batch_size=64 resume_from_checkpoint_id=4xnuhlo8
