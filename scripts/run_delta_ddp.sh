#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --partition=ghx4
#SBATCH --account=bebe-dtai-gh
#SBATCH --gpu-bind=verbose,closest
#SBATCH --out=logs/%j.log
#SBATCH --error=logs/%j.log

# Usage:
# cd polygloss
# module use /sw/user/modules/python
# module load python/miniforge3_pytorch
# conda activate polygloss
# sbatch scripts/run_delta_ddp.sh <path_to_config.cfg> (--monoling)

# To create the env see here: https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/python/pytorch.html#pytorch-pip-install

echo "=== CUDA + PyTorch diagnostics ==="
which python
python - <<'PY'
import torch, os
print("CUDA visible devices:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Torch CUDA version:", torch.version.cuda)
print("Torch built with:", torch.__config__.show())
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Detected GPUs:", torch.cuda.device_count())
    print("GPU 0:", torch.cuda.get_device_name(0))
PY

module list

export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

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
