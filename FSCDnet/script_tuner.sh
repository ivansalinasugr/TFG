#!/bin/bash

#SBATCH --job-name F35_tuner
#SBATCH --partition dios
#SBATCH --gres=gpu:1
#SBATCH --output=F35_tuner_res.out
#SBATCH  -w dionisio
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/isalinas/envs/python3.11
export TFHUB_CACHE_DIR=.

focal=35

python ./SCDnet/codigos/SCD_MainTuner.py --src_root /mnt/homeGPU/isalinas/FSCDnet/SCDnet --focal $focal --n_epochs 10 > ./F${focal}_tuner.txt
