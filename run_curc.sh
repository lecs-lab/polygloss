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

set -x

module purge
module load miniforge
mamba activate polygloss
cd "/projects/$USER/polygloss"

if [[ "$2" == "--monoling" ]]; then
    echo "Running multiple monolingual experiments"

    for glottocode in arap1274 gitx1241 lezg1247 natu1246 nyan1302 dido1241 uspa1245
    do
        echo "Running with $glottocode"
        python run.py "$1" --overrides "glottocode=$glottocode"
    done
else
    python run.py "$1"
fi
