#!/bin/bash
set -e
OUTDIR="build"
INPUTS="inputs/behavior_data.tbl"

mkdir -p ${OUTDIR}
awk 'NR>1 {gsub(/[0-9]/, "", $2); print $1"_"$2}' inputs/behavior_data.tbl | sort | uniq | \
    parallel Rscript scripts/ssm-training.R {}

