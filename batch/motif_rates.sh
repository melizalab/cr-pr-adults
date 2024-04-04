#!/bin/bash
set -e
OUTDIR="build/"
PYTHON=venv/bin/python3

if [[ -t 0 ]]; then
    echo "usage: batch/motif_rates.sh < list_of_units"
    exit 1
fi
mkdir -p ${OUTDIR}
parallel ${PYTHON} scripts/motif_rates.py -o ${OUTDIR} {}
