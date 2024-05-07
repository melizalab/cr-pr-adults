#!/usr/bin/env python
# -*- mode: python -*-
"""Compute features from average spike waveforms """
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import quickspikes as qs
from dlab import nbank, spikes
from sklearn.mixture import GaussianMixture

upsampled_rate_khz = 150
ms_before = 2.0
ms_after = 3.0
stdev_after_i = int(2.3 * upsampled_rate_khz)
max_trough_z = -15
max_trough_v = -400
max_peak2_t = 1.5


def waveform_features(x):
    x_baseline = (x.iloc[10].sum() + x.iloc[-10].sum()) / 20
    sign = np.sign(x.max() + x.min() - x_baseline)
    if sign > 0:
        return pd.Series({"sign": sign})
    trough_i = x.idxmin()
    trough_x = x.loc[trough_i]
    peak1_i = x.loc[:trough_i].idxmax()
    peak2_i = x.loc[trough_i:].idxmax()
    return pd.Series(
        {
            "sign": sign,
            "trough_v": trough_x - x_baseline,
            "peak1_t": (peak1_i - trough_i) / upsampled_rate_khz,
            "peak1_v": x.loc[peak1_i] - x_baseline,
            "peak2_t": (peak2_i - trough_i) / upsampled_rate_khz,
            "peak2_v": x.loc[peak2_i] - x_baseline,
            "sd_v": x.loc[stdev_after_i:].std(),
        }
    )


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug", help="show verbose log messages", action="store_true"
    )
    nbank.add_registry_argument(parser)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="if set, search this directory for spike waveform files before the registry",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path.cwd(),
        help="the directory to store output files (default %(default)s)",
    )
    parser.add_argument(
        "unitfile", type=Path, help="file with list of units to analyze"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        format="%(message)s", level=logging.DEBUG if args.debug else logging.INFO
    )
    with open(args.unitfile) as fp:
        unit_names = [clean for line in fp if len(clean := line.strip()) > 0]

    waveforms = []
    for resource_name, path in nbank.find_resources(
        *(f"{unit_name}_spikes" for unit_name in unit_names),
        alt_base=args.data_dir,
    ):
        unit_name = resource_name.rstrip("_spikes")
        if not isinstance(path, Path):
            logging.info(
                "  - %s: waveform resource %s not found",
                unit_name,
                resource_name,
            )
        else:
            data = spikes.load_waveforms(path)
            i_before = data.peak_index - int(ms_before * data.sampling_rate / 1000)
            i_after = data.peak_index + int(ms_after * data.sampling_rate / 1000)
            all_spikes = data.waveforms[:, i_before:i_after]
            nspikes, npoints = all_spikes.shape
            logging.info(
                "  - %s: loaded %d waveforms from %s", unit_name, nspikes, path
            )
            upsampled = qs.tools.fftresample(
                all_spikes,
                int((ms_before + ms_after) * upsampled_rate_khz),
                reflect=True,
            )
            mean_spike = pd.Series(
                upsampled.mean(0),
                index=pd.RangeIndex(
                    int(-ms_before * upsampled_rate_khz),
                    int(ms_after * upsampled_rate_khz),
                ),
                name=unit_name,
            )
            waveforms.append(mean_spike)

    waveforms = pd.concat(waveforms, axis=1)
    waveform_file = args.output_dir / "mean_spike_waveforms.csv"
    waveforms.to_csv(waveform_file, index_label="time_samples")
    logging.info("- wrote mean waveforms to %s", waveform_file)

    logging.info("- computing waveform features")
    features = waveforms.apply(waveform_features).T
    features["ptratio"] = -features.peak2_v / features.trough_v
    too_small = ((features.trough_v / features.sd_v) > max_trough_z) & (
        features.trough_v > max_trough_v
    )
    positive = features.sign >= 0
    too_long = features.peak2_t > max_peak2_t
    excluded = features.index[too_small | positive | too_long]
    included = features.index.difference(excluded)

    logging.info("- clustering waveforms (%d excluded)", len(excluded))
    X = features.loc[included][["peak2_t", "ptratio"]]
    gmix = GaussianMixture(n_components=2).fit(X)
    narrow = gmix.means_[:, 0].argmin()
    is_narrow = pd.Series(
        pd.cut(1.0 * (gmix.predict(X) == narrow), bins=2, labels=("wide", "narrow")),
        index=included,
        name="spike",
    )
    features = features.join(
        pd.Series(is_narrow, index=included, name="spike"), how="left"
    )

    feature_file = args.output_dir / "mean_spike_features.csv"
    features.to_csv(feature_file, index_label="unit")
    logging.info("- wrote waveform features to %s", feature_file)


if __name__ == "__main__":
    main()
