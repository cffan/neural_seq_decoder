#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --job-name=rescore
#SBATCH --mail-type=ALL
#SBATCH --mem=400GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --partition=henderj,owners
#SBATCH --signal=USR1@120
#SBATCH --time=2880
#SBATCH --constraint=[GPU_MEM:32GB|GPU_MEM:40GB|GPU_MEM:80GB]

ml gcc/10.1.0
ml load cudnn/8.6.0.163
ml load cuda/11.7.1

python eval_competition.py --modelPath=$1
