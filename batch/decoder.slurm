#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32000
#SBATCH -t 04:00:00
#SBATCH -o logfiles/decoder_%a.log
#SBATCH --array 1-2902
#SBATCH -p standard
#SBATCH -A melizalab
set -e
module load miniforge/24.3.0-py3.11
source activate cr-pr-adults
PYTHON=${HOME}/.conda/envs/cr-pr-adults/bin/python3

OUTDIR="build/"
mkdir -p ${OUTDIR}

DATASET_DIR="datasets/zebf-social-acoustical-ephys"
CONTROL_FILE="inputs/decoder_tasks.txt"
sed -n "${SLURM_ARRAY_TASK_ID}"p ${CONTROL_FILE} | { read UNIT_LIST N_UNITS SEED
${PYTHON} scripts/decoder.py ${OPTIONS} --n-units ${N_UNITS} --random-seed ${SEED} -o ${OUTDIR} --pprox-dir ${DATASET_DIR}/responses --metadata-dir ${DATASET_DIR}/metadata --stim-dir ${DATASET_DIR}/stimuli ${UNIT_LIST}
}
