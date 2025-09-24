#!/bin/bash
#SBATCH --account=naiss2024-22-1529
#SBATCH --gpus-per-node=V100:1
#SBATCH --time=02:30:00

apptainer exec ../../danse/container.sif python3 train.py -c lorenz63.json