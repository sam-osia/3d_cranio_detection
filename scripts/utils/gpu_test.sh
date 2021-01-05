#!/bin/bash


# configure all the slurm stuff:
#SBATCH --job-name=test_job
#SBATCH --output=/h/samosia/Git/3d_cranio_detection/cluster_logs/output-%N-%j.out
#SBATCH --error=/h/samosia/Git/3d_cranio_detection/cluster_logs/error-%N-%j.out
#SBATCH --open-mode=append
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1


# activate project virtualenv
source /h/samosia/Git/3d_cranio_detection/venv/bin/activate

# setup environmental variables to point to the correct CUDA build
export PATH=/pkgs/cuda-10.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/pkgs/cuda-10.1/lib64:/pkgs/cudnn-10.1-v7.6.3.30/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# run the script
python ./gpu_test.py >> /h/samosia/Git/3d_cranio_detection/cluster_logs/gpu_test.log

deactivate
