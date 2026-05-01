#!/bin/bash
#SBATCH -p polariton
#SBATCH --output=qRemove.log
#SBATCH --mem-per-cpu=10GB
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

OUTDIR="$SLURM_SUBMIT_DIR/Data_Laser3"
echo "Removing output directory: $OUTDIR"

# delete A_* files
find "$OUTDIR" -xdev -type f -name 'psiF_*' -print0 | xargs -0 -r rm -f & pidA=$!

# delete B_* files
find "$OUTDIR" -xdev -type f -name 'psiB_*' -print0 | xargs -0 -r rm -f & pidB=$!

wait $pidA $pidB

# clean empty dirs
find "$OUTDIR" -xdev -depth -type d -empty -delete
mkdir -p "$OUTDIR"
echo "Done."