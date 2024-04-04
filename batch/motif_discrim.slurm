#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000
#SBATCH -t 00:60:00
#SBATCH -o logfiles/motif_discrim_%a.log
#SBATCH --array 1-2954
#SBATCH -p standard
#SBATCH -A melizalab
set -e
module load anaconda/2023.07-py3.11
source activate noise-invariant-analysis
PYTHON=${HOME}/.conda/envs/noise-invariant-analysis/bin/python3

OUTDIR="build/"
UNITLIST="inputs/all_units.txt"
UNIT=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${UNITLIST})
echo -e "- computing motif discrimination for ${UNIT}"

mkdir -p ${OUTDIR}
${PYTHON} scripts/motif_discrim.py --shuffle-replicates 500 -o ${OUTDIR} ${UNIT}
