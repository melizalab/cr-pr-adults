#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Compute average firing rates for responses to motifs"""
import json
import logging
from pathlib import Path
import sys

import pandas as pd
from core import MotifBackgroundSplitter, split_trials
from dlab import nbank


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug", help="show verbose log messages", action="store_true"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help="the directory to store output files (default %(default)s)",
    )
    parser.add_argument("unit", help="name of the unit to analyze")
    args = parser.parse_args(argv)
    logging.basicConfig(
        format="%(message)s", level=logging.DEBUG if args.debug else logging.INFO
    )

    pprox_file = nbank.find_resource(args.unit)
    logging.info("- reading spikes from %s", pprox_file)
    unit = json.loads(pprox_file.read_text())
    logging.info("- splitting trials by motif")
    motifs = (
        split_trials(MotifBackgroundSplitter(), unit)
        .reset_index()
        .rename(columns=lambda s: s.replace("-", "_"))
        .set_index(["background_dBFS", "foreground_dBFS", "source_trial", "foreground"])
    )

    # compute rates
    motifs["n_events"] = motifs.events.fillna("").apply(len)
    motifs["rate"] = motifs.n_events / motifs.interval_end
    # compute response strength by subtracting off silence
    rs = (
        motifs.rate.unstack()
        .apply(lambda x: x - x.silence, axis=1)
        .stack()
        .rename("rs")
    )
    # merge into a data table and add a column for unit
    cols_to_keep = [
        "foreground",
        "source_trial",
        "interval",
        "background_dBFS",
        "interval_end",
        "n_events",
        "rate",
        "rs",
    ]
    data = pd.merge(motifs, rs, left_index=True, right_index=True).reset_index()[
        cols_to_keep
    ]
    if args.output_dir is not None:
        outfile = args.output_dir / f"{args.unit}_rates.csv"
        data.insert(0, "unit", args.unit)
        logging.info("- wrote csv to %s", outfile)
    else:
        outfile = sys.stdout
    data.to_csv(outfile, index=False)


if __name__ == "__main__":
    main()
