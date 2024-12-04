#!/usr/bin/env bash
set -e
OUTDIR="datasets"

# retrieve the datasets

# check file integrity
shasum -a 512 -c ${OUTDIR}/sha512sums

# unpack the datasets
for zipfile in ${OUTDIR}/*.zip; do
    if [ -f "$zipfile" ] ; then
	unzip -o ${zipfile} -d ${OUTDIR}
    fi
done

# delete the zip files
