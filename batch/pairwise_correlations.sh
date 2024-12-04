#!/usr/bin/env bash
set -e
OUTDIR="build/"
: ${PYTHON:=venv/bin/python3}

if [[ -t 0 ]]; then
    echo "usage: batch/pairwise_correlations.sh < table_of_recordings"
    exit 1
fi
mkdir -p ${OUTDIR}
parallel --colsep ',' --skip-first-line ${PYTHON} scripts/pairwise_correlations.py --data-dir ${OUTDIR} {1}
