#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32000m
#SBATCH --time=60:00
#SBATCH --account=zhengya98
#SBATCH --partition=gpu_mig40
#SBATCH --gpus-per-node=1

module load python/3.10.4

cd /scratch/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
source .venv/bin/activate

cd library/retroinfer

pip install --no-build-isolation .
