#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000
#SBATCH -t 00:20:00
#SBATCH -o logfiles/motif_discrim_%a.log
#SBATCH --array 1-2954
#SBATCH -p standard
#SBATCH -A melizalab
set -e
module load miniforge/24.3.0-py3.11
source activate cr-pr-adults
PYTHON=${HOME}/.conda/envs/cr-pr-adults/bin/python3

DATASETDIR="datasets/zebf-social-acoustical-ephys"
OUTDIR="build/"
UNITLIST="inputs/all_units.tbl"
UNIT=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${UNITLIST})
echo -e "- computing motif discrimination for ${UNIT}"

mkdir -p ${OUTDIR}
${PYTHON} scripts/motif_discrim.py --shuffle-replicates 500 -m ${DATASETDIR}/metadata -o ${OUTDIR} ${DATASETDIR}/responses/${UNIT}.pprox
