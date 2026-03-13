#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
##SBATCH --qos=preemptable
##SBATCH --gres=gpu:a100:1
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
##SBATCH --partition=blanca-kann
#SBATCH --account=blanca-curc-gpu
#SBATCH --gres=gpu:1          
#SBATCH --out=logs/uspanteko%j.log
#SBATCH --error=logs/uspanteko%j.log
#SBATCH --array=0-2

#5 languages * 3 seeds = 15 (0-14)
module purge
module load miniforge 
mamba activate polygloss2
cd "/projects/$USER/fresh-polygloss"

#LANGUAGES=("lezg1247")

LANGUAGES=("uspa1245")

SEEDS=(42 42 44)
NUM_SEEDS=${#SEEDS[@]}

# --- LOGIC FOR MAPPING ARRAY ID TO CONFIG ---
# Determine Language Index (Integer Division)
LANG_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))
LANGUAGE=${LANGUAGES[$LANG_IDX]}

# Determine Limit Index (Modulo)
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))
SEED=${SEEDS[$SEED_IDX]}


python active_learning.py experiments/byt5_active_learning_seeds/train_active_byt5.cfg --overrides glottocode=$LANGUAGE dataset_key="stupidfishlady/sigmorphon_st" seed=$SEED
