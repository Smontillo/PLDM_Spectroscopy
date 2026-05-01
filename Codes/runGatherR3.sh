#!/bin/bash
#SBATCH -p polariton
#SBATCH --job-name=R3t         # create a name for your job
#SBATCH --ntasks=50               # total number of tasks
#SBATCH --cpus-per-task=1          # cpu-cores per task
#SBATCH --mem-per-cpu=1G           # memory per cpu-core
#SBATCH -t 1-00:00:00              # total run time limit (HH:MM:SS)
#SBATCH --output=Plot.out
#SBATCH --error=Plot.err

mpiexec -n 50 python ConstructResponseF.py $1
