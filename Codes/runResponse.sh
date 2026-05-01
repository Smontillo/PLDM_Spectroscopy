#!/bin/bash
#SBATCH -p polariton #preempt
#SBATCH --output=qDyn.log
#SBATCH --mem-per-cpu=10GB
#SBATCH -t 5-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

python ConstructResponseF2.py $1
# NUMBA_DEBUG=1 python -u Dynamics.py
