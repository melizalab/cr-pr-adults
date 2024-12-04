#!/usr/bin/env bash
set -e
OUTDIR="build/"
: ${PYTHON:=venv/bin/python3}

if [[ -t 0 ]]; then
    echo "usage: batch/motif_discrim.sh < list_of_units"
    exit 1
fi
mkdir -p ${OUTDIR}
parallel ${PYTHON} scripts/motif_discrim.py --shuffle-replicates 500 -m datasets/zebf-social-acoustical-ephys/metadata -o ${OUTDIR} datasets/zebf-social-acoustical-ephys/responses/{}.pprox
