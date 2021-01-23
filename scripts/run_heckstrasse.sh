#!/bin/bash
#
#SBATCH --job-name=heckstrasse
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=2G

#SBATCH --gres=gpu:4

source /opt/miniconda/etc/profile.d/conda.sh
source /home/bgye/dt/precog/precog_env.sh
conda activate $PRECOGCONDAENV

python /home/bgye/dt/precog/precog/esp_train.py
