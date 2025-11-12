#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
#SBATCH --partition=ghx4
#SBATCH --account=mginn
#SBATCH --gpu-bind=verbose,closest
#SBATCH --out=logs/%j.log
#SBATCH --error=logs/%j.log

# Usage: sbatch run_curc_ddp.sh <path_to_config.cfg> (--monoling)

export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

module load python/miniforge3_pytorch
conda activate polygloss
cd "/projects/bebe/$USER/polygloss"

set -x

if [[ "$2" == "--monoling" ]]; then
    echo "Running multiple monolingual experiments"

    for glottocode in ainu1240 ruul1235 lezg1247 natu1246 nyan1302 dido1241 uspa1245 arap1274
    do
        echo "Running with $glottocode"
        torchrun \
            --nproc_per_node=$SLURM_NTASKS_PER_NODE \
            --nnodes=$SLURM_NNODES \
            --node_rank=$SLURM_NODEID \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            run.py "$1" --overrides glottocode=$glottocode
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
