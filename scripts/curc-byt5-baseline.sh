#!/bin/bash
#SBATCH --nodes=1           # Number of requested nodes
#SBATCH --gres=gpu
#SBATCH --ntasks=4          # Number of requested cores
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00          # Max walltime              # Specify QOS
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --account=blanca-curc-gpu
#SBATCH --out=finetune_glosslm.%j.out      # Output file name
#SBATCH --error=finetune_glosslm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.ginn@colorado.edu

# purge all existing modules
module purge
# Load the python module
module load anaconda
# Run Python Script
conda activate AutoIGT
cd "/projects/migi8081/glosslm/src"

for lang in arap1274 dido1241 uspa1245 nyan1302 natu1246 lezg1247
do
  python3 pretrain_multilingual_model.py --mode finetune --exp_name byt5-baseline-${lang} --ft_glottocode ${lang} --output_model_path ../models/byt5-baseline-${lang} --max_epochs 20 --pretrained_model "google/byt5-base"
done