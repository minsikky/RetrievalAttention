#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16000m
#SBATCH --time=30:00
#SBATCH --account=zhengya98
#SBATCH --partition=largemem
#SBATCH --gpus-per-node=0

module load python/3.10.4

cd /scratch/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
.venv/bin/pip install -r requirements.txt
