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
#SBATCH --out=logs/%j.log
#SBATCH --error=logs/%j.log

export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

module purge
module load miniforge
mamba activate polygloss
cd "/projects/$USER/polygloss"

set -x

if [[ "$2" == "--monoling" ]]; then
    echo "Running multiple monolingual experiments"

    for glottocode in arap1274 gitx1241 lezg1247 natu1246 nyan1302 dido1241 uspa1245
    do
        echo "Running with $glottocode"
        torchrun \
            --nproc_per_node=$SLURM_NTASKS_PER_NODE \
            --nnodes=$SLURM_NNODES \
            --node_rank=$SLURM_NODEID \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            run.py "$1" --overrides "glottocode=$glottocode"
    done
else
    torchrun \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    run.py "$1"
fi
