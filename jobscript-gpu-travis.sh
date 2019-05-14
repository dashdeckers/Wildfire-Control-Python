#!/bin/bash
#SBATCH --job-name=DQN
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

pwd
module purge
source ../FireFighter/bin/activate
python main.py -headless

