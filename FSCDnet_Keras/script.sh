#!/bin/bash

#SBATCH --job-name F27
#SBATCH --partition dios
#SBATCH --gres=gpu:1
#SBATCH --output=F27res.out
#SBATCH  -w atenea
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate tf1.15py36
export TFHUB_CACHE_DIR=.

python ./SCDnet/SCD_Main.py > ./F27.txt
