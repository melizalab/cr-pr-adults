#!/usr/bin/env python
# -*- mode: python -*-
"""Compute pairwise correlations for all units in a site"""
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd


def summarize_stim(df):
    """Summarize responses of a pair to a stimulus

    df: trials x 2 array
    """
    n_trials = df.shape[0]
    avgs = df.values.mean(0)
    cent = df.values - avgs
    denom = np.sqrt(np.sum(cent**2, 0).prod())
    if denom == 0:
        noise_corr = shift_corr = np.nan
    else:
        noise_corr = np.sum(cent[:, 0] * cent[:, 1]) / denom
        # shift_corr = np.sum(cent[:, 0] * np.roll(cent[:, 1], 1)) / denom
        # compute using all the shifts
        shift_corr = [
            np.sum(cent[:, 0] * np.roll(cent[:, 1], i)) / denom
            for i in range(1, n_trials)
        ]
    return pd.Series(
        [*avgs, noise_corr, np.mean(shift_corr)],
        index=["avg_1", "avg_2", "noise", "shifted"],
    )


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug", help="show verbose log messages", action="store_true"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default="build",
        help="specify directory where _rates.csv files are, and where to store output",
    )
    parser.add_argument("site", help="name of the site")
    args = parser.parse_args(argv)
    logging.basicConfig(
        format="%(message)s", level=logging.DEBUG if args.debug else logging.INFO
    )
    logging.info("- loading rate data from %s", args.data_dir)
    unit_files = args.data_dir.glob(f"{args.site}*_rates.csv")
    responses = (
        pd.concat([pd.read_csv(f) for f in unit_files])
        .query("foreground != 'background'")
        .query("background_dBFS == -100 | foreground == 'silence'")
        .set_index(["foreground", "source_trial", "unit"])
        # .query("foreground not in ('silence', 'background')")
        # .query("background_dBFS == -100")
        # .set_index(["foreground", "interval", "unit"])
        .loc[:, "rate"]
        .unstack()
    )
    logging.info("- loaded data from %s units", len(responses.columns))

    outfile = args.data_dir / f"{args.site}_correlations.csv"
    logging.info("- output to %s", outfile)
    with open(outfile, "w") as fp:
        print(
            "unit_1,unit_2,signal,evoked_noise,evoked_shifted,evoked_noise_c,spont_noise,spont_shifted,spont_noise_c",
            file=fp,
        )
        for pair in itertools.combinations(responses.columns, 2):
            logging.debug("  - analyzing %s", pair)
            df = responses.loc[:, pair]
            by_stim = df.groupby(level=0).apply(summarize_stim)
            evoked = by_stim.drop("silence")
            spont = by_stim.loc["silence"]
            stim_mean = evoked[["avg_1", "avg_2"]].mean()
            if any(stim_mean < 0.001):
                signal_corr = np.nan
            else:
                signal_corr = by_stim.avg_1.corr(by_stim.avg_2)
            print(
                f"{pair[0]},{pair[1]},{signal_corr:.4f},{evoked.noise.mean():.4f},{evoked.shifted.mean():.4f},{(by_stim.noise - by_stim.shifted).mean():.4f},{spont.noise:.4f},{spont.shifted:.4f},{spont.noise - spont.shifted:.4f}",
                file=fp,
            )


if __name__ == "__main__":
    main()
