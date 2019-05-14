#!/bin/bash
#SBATCH --job-name=DQN
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --partition=regular

pwd
module purge
make
source ../FireFighter/bin/activate
python main.py -headless

