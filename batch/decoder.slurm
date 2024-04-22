#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64000
#SBATCH -t 04:00:00
#SBATCH -o logfiles/decoder_%a.log
#SBATCH -p standard
#SBATCH -A melizalab
set -e
module load anaconda/2023.07-py3.11
source activate cr-pr-adults
PYTHON=${HOME}/.conda/envs/cr-pr-adults/bin/python3

OUTDIR="build/"
UNIT_LIST="build/cr_units.txt"
N_UNITS=1055
SEED=${SLURM_ARRAY_TASK_ID}

mkdir -p ${OUTDIR}
${PYTHON} scripts/decoder.py ${OPTIONS} --n-units ${N_UNITS} --random-seed ${SEED} -o ${OUTDIR} ${UNIT_LIST}
