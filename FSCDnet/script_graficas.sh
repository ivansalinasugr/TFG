#!/bin/bash

#SBATCH --job-name metricas
#SBATCH --partition dios
#SBATCH --gres=gpu:1
#SBATCH --output=metricas_res.out
#SBATCH  -w dionisio

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/isalinas/envs/python3.11
export TFHUB_CACHE_DIR=.

python ./SCDnet/codigos/SCD_MainGraficas.py --src_root /mnt/homeGPU/isalinas/FSCDnet/SCDnet > ./metricas.txt