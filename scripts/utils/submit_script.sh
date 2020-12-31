#!/bin/bash

dirname=$1
abs_dirname=$(cd "$(dirname "$1")"; pwd)/$(basename "$1")
echo hello
echo $dirname
echo $abs_dirname


## configure all the slurm stuff:
##SBATCH --job-name=test_job
##SBATCH --output=/h/samosia/Git/endomondo_analysis/logs/test_logs/output-%N-%j.out
##SBATCH --error=/h/samosia/Git/endomondo_analysis/logs/test_logs/error-%N-%j.out
##SBATCH --open-mode=append
##SBATCH --partition=gpu
##SBATCH --ntasks=4
##SBATCH --cpus-per-task=4
##SBATCH --mem=32G
##SBATCH --gres=gpu:1
##SBATCH --nodes=1
#
## activate project virtualenv
#source /h/samosia/python_envs/tf2/bin/activate
#
## setup environmental variables to point to the correct CUDA build
#export PATH=/pkgs/cuda-10.1/bin${PATH:+:${PATH}}
#export LD_LIBRARY_PATH=/pkgs/cuda-10.1/lib64:/pkgs/cudnn-10.1-v7.6.3.30/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
#
## run the script
#python /h/samosia/Git/endomondo_analysis/scripts/more_complicated_model.py >> /h/samosia/Git/endomondo_analysis/logs/test_logs/more_complicated_model.log
#
#deactivate
