#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96000
#SBATCH -t 05:00:00
#SBATCH -o logfiles/ssm-training-%a.log
#SBATCH --array 0-33
#SBATCH -p standard
#SBATCH -A melizalab
set -e
module load gcc/11.4.0 openmpi/4.1.4 R/4.3.1

DATASETS=("C269_pretrain" "C269_train" \
	  "C280_pretrain" "C280_train" \
	  "C291_pretrain" "C291_train" \
	  "C293_pretrain" "C293_train" \
	  "C294_pretrain" "C294_train" \
	  "C235_pretrain" "C235_train" \
	  "C250_pretrain" \
	  "C193_pretrain" "C193_train" \
	  "C191_pretrain" "C191_train" \
	  "C197_pretrain" "C197_train" \
	  "Rb255_pretrain" "Rb255_train" \
	  "Rb266_pretrain" "Rb266_train" \
	  "Rb256_pretrain" "Rb256_train" \
	  "Rb272_pretrain" "Rb272_train" \
	  "Rb276_pretrain" "Rb276_train" \
	  "Rb279_pretrain" "Rb279_train" \
	  "Rb266_pretrain" "Rb266_train" \
	  "Rb284_pretrain" \
	  "Rb299_pretrain")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
echo -e "Computing state-space model for ${DATASET}:"

Rscript scripts/ssm-pretraining.R ${DATASET}
