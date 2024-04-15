#!/bin/bash
set -e
OUTDIR="build"
PYTHON=venv/bin/python3
DECIDE_HOST=http://pholia.lab:4000/decide/api/

if [[ -t 0 ]]; then
    echo "usage: batch/retrieve_trials.sh < table_of_behavior_experiments"
    exit 1
fi
mkdir -p ${OUTDIR}
parallel --colsep ' +' --skip-first-line -j4 ${PYTHON} -m decide_analysis.get_data -r ${DECIDE_HOST} --fields id,subject,time,trial,stimulus,lights,correct,response,result,correction,experiment --name gng -k experiment__startswith={3} --output ${OUTDIR}/{1}_{2}_trials.csv {1}
