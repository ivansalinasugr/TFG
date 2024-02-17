#!/bin/bash

#SBATCH --job-name F27
#SBATCH --partition dios
#SBATCH --gres=gpu:1
#SBATCH --output=F27res.out
#SBATCH  -w atenea
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/isalinas/envs/devfacialnet
export TFHUB_CACHE_DIR=.

python ./SCDnet/codigos/SCD_Main.py --src_root /mnt/homeGPU/FSCDnet/SCDnet --focal 27 --n_epochs 10 --run_name pytorchGPU > ./F27.txt
