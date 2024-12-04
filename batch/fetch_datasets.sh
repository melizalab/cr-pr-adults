#!/usr/bin/env bash
set -e
OUTDIR="datasets/"

# retrieve the datasets

# check file integrity
shasum -a 512 -c ${OUTDIR}sha512sums

# unpack the datasets

# delete the zip files
