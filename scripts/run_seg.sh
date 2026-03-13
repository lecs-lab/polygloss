#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
##SBATCH --qos=preemptable
##SBATCH --gres=gpu:a100:1 
#SBATCH --qos=blanca-curc-gpu
#SBATCH --account=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --gres=gpu:1                  
#SBATCH --out=logs/segmentation%j.log
#SBATCH --error=logs/segmentation%j.log
#SBATCH --array=0-5

#6 languages * 1 limits = 6 (0-5)
module purge
module load miniforge 
mamba activate polygloss2
cd "/projects/$USER/fresh-polygloss"


LANGUAGES=("arap1274" "natu1246" "nyan1302" "lezg1247" "dido1241" "gitx1241")


# --- LOGIC FOR MAPPING ARRAY ID TO CONFIG ---
# Determine Language Index (Integer Division)
LANG_IDX=$SLURM_ARRAY_TASK_ID 
LANGUAGE=${LANGUAGES[$LANG_IDX]}


python run.py experiments/segmentation/train_segmentation.cfg --overrides glottocode=$LANGUAGE dataset_key="stupidfishlady/sigmorphon_st" 
