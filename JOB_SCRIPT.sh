#!/bin/bash -l
#SBATCH --job-name=clr
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

#SBATCH --partition=gpu
#SBATCH --gres gpu:1
#SBATCH -C v100

#SBATCH -n 1
#SBATCH -c 12
#SBATCH --time=04:00:00
#SBATCH --mem=16G

module load container_env pytorch-gpu/2.8.0

if [ -z "$1" ]; then
    echo "Error: no config file provided. Usage: sbatch JOB_SCRIPT.sh <config>"
    exit 1
fi

crun -p ../envs/clr python experiment.py $1
