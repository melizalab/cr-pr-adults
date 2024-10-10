#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96000
#SBATCH -t 05:00:00
#SBATCH -o logfiles/ssm-training-%a.log
#SBATCH --array 1-36
#SBATCH -p standard
#SBATCH -A melizalab
set -e
module load gcc/11.4.0 openmpi/4.1.4 R/4.3.1

# this will pick out the unique combinations of <animal>_<phase> in behavior_data.tbl
awk 'NR>1 {gsub(/[0-9]/, "", $2); print $1"_"$2}' inputs/behavior_data.tbl | sort | uniq | grep -v probe | sed -n "${SLURM_ARRAY_TASK_ID}p" | \
    xargs Rscript scripts/ssm-training.R
