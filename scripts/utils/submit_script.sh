#!/bin/bash


# configure all the slurm stuff:
#SBATCH --job-name=test_job
#SBATCH --output=/h/samosia/Git/3d_cranio_detection/cluster_logs/output-%N-%j.out
#SBATCH --error=/h/samosia/Git/3d_cranio_detection/cluster_logs/error-%N-%j.out
#SBATCH --open-mode=append
#SBATCH --partition=gpu
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -S|--script)
    SCRIPT_REL_DIR="$2"
    shift # past argument
    shift # past value
    ;;
    -R|--run_id)
    RUN_ID="$2"
    shift # past argument
    shift # past value
    ;;
    -T|--tag)
    TAG="$2"
    shift # past argument
    shift # past value
    ;;
    --default)
    DEFAULT=YES
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

SCRIPT_DIR=$(cd "$(dirname "$SCRIPT_REL_DIR")"; pwd)/$(basename "$SCRIPT_REL_DIR")

echo "${POSITIONAL[@]}"
echo "RELATIVE DIR: ${SCRIPT_REL_DIR}"
echo "ABSOLUTE DIR: ${SCRIPT_DIR}"
echo "RUN_ID: ${RUN_ID}"
echo "TAG: ${TAG}"


## activate project virtualenv
#source /h/samosia/Git/3d_cranio_detection/venv/bin/activate
#
## setup environmental variables to point to the correct CUDA build
#export PATH=/pkgs/cuda-10.1/bin${PATH:+:${PATH}}
#export LD_LIBRARY_PATH=/pkgs/cuda-10.1/lib64:/pkgs/cudnn-10.1-v7.6.3.30/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
#
## run the script
#echo "python $script_dir $commands >> /h/samosia/Git/3d_cranio_detection/cluster_logs/$commands.log"
#
#deactivate
