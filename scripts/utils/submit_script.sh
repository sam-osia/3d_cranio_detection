#!/bin/bash

# configure all the slurm stuff:
#SBATCH --job-name=test_job
#SBATCH --output=/h/samosia/Git/endomondo_analysis/logs/test_logs/output-%N-%j.out
#SBATCH --error=/h/samosia/Git/endomondo_analysis/logs/test_logs/error-%N-%j.out
#SBATCH --open-mode=append
#SBATCH --partition=gpu
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

