#!/bin/bash
#SBATCH -p polariton #preempt
#SBATCH --output=qPlot.log
#SBATCH --mem-per-cpu=10GB
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

python PlotPrev.py
# python -u Dynamics.py
