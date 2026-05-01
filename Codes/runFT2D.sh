#!/bin/bash
#SBATCH -p polariton
#SBATCH --output=qFT2D.log
#SBATCH --mem-per-cpu=10GB
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

# python Dynamics.py
python -u Plot_2D.py $1
