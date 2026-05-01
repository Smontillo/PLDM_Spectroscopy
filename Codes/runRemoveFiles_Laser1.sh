#!/bin/bash
#SBATCH -p polariton
#SBATCH --output=qRemove.log
#SBATCH --mem-per-cpu=10GB
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

OUTDIR="$SLURM_SUBMIT_DIR/Data_Laser1"

echo "Removing output directory: $OUTDIR"
rm -rf --one-file-system -- "$OUTDIR"
mkdir -p "$OUTDIR"
echo "Done removing output directory: $OUTDIR"


