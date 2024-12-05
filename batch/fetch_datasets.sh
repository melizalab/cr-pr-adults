#!/usr/bin/env bash
set -e
OUTDIR="datasets"

# retrieve the datasets
curl -o ${OUTDIR}/zebf-social-acoustical-stats.zip https://figshare.com/ndownloader/files/50976225
curl -o ${OUTDIR}/zebf-discrim-noise.zip https://figshare.com/ndownloader/files/50974560
curl -o ${OUTDIR}/zebf-social-acoustical-ephys.zip https://figshare.com/ndownloader/files/50976219

# check file integrity
shasum -a 512 -c ${OUTDIR}/sha512sums

# unpack the datasets
for zipfile in ${OUTDIR}/*.zip; do
    if [ -f "$zipfile" ] ; then
	unzip -o ${zipfile} -d ${OUTDIR}
    fi
done

# delete the zip files
rm ${OUTDIR}/*.zip
