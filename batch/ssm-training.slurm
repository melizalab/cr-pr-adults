#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96000
#SBATCH -t 05:00:00
#SBATCH -o logfiles/ssm-training.log
#SBATCH -p standard
#SBATCH -A melizalab
set -e
module load gcc/11.4.0 openmpi/4.1.4 R/4.3.1
Rscript scripts/ssm-training.R datasets/zebf-discrim-noise/trials/C280_train_trials.csv
