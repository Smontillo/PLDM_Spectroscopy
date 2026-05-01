#!/bin/bash
#SBATCH -p polariton
#SBATCH --output=qDyn.log
#SBATCH --mem-per-cpu=10GB
#SBATCH -t 5-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

python Dynamics.py
# NUMBA_DEBUG=1 python -u Dynamics.py
