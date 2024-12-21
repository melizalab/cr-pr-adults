#!/usr/bin/env python
# -*- mode: python -*-
"""Compute average firing rates for responses to motifs"""
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from core import MotifBackgroundSplitter, split_trials, find_resource

clean_dBFS = -100


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
        help="the directory to store output file (if not set, outputs to standard out)",
    )
    parser.add_argument(
        "--metadata-dir",
        "-m",
        type=Path,
        help="the directory where stimulus metadata files are stored",
    )
    parser.add_argument("unit", type=Path, help="pprox data (name of unit or path to file) to analyze")
    args = parser.parse_args(argv)
    logging.basicConfig(
        format="%(message)s", level=logging.DEBUG if args.debug else logging.INFO
    )

    if args.unit.exists():
        pprox_file = args.unit
    else:
        pprox_file = find_resource(str(args.unit))
    logging.info("- reading spikes from %s", pprox_file)
    unit = json.loads(pprox_file.read_text())
    logging.info("- splitting trials by motif")
    motifs = (
        split_trials(MotifBackgroundSplitter(), unit, args.metadata_dir)
        .reset_index()
        .rename(columns=lambda s: s.replace("-", "_"))
        .set_index(["background_dBFS", "source_trial", "foreground"])
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
        unit_name = args.unit.stem
        outfile = args.output_dir / f"{unit_name}_rates.csv"
        data.insert(0, "unit", unit_name)
        logging.info("- wrote csv to %s", outfile)
    else:
        outfile = sys.stdout
    data.to_csv(outfile, index=False)

    # fit a GLM to the counts to estimate significance - not used because we
    # can't calculate marginal means
    # pooled = (
    #     data.query("foreground == 'silence' | background_dBFS==-100")
    #     .query("foreground != 'background'")
    #     .groupby("foreground")
    #     .aggregate({"n_events": ["sum"], "interval_end": ["sum"]})
    #     .droplevel(-1, axis=1)
    # )
    # # ensure there is one spike in the silence condition
    # pooled.loc["silence", "n_events"] = max(pooled.loc["silence", "n_events"], 1)
    # motif_names = ["silence"] + list(set(pooled.index.unique()) - {"silence"})
    # pooled = pooled.reset_index()
    # pooled["foreground"] = pd.Categorical(
    #     pooled.foreground, categories=motif_names, ordered=True
    # )
    # lm = smf.glm(
    #     "n_events ~ foreground",
    #     data=pooled,
    #     family=sm.families.Poisson(),
    #     offset=np.log(pooled["interval_end"]),
    # ).fit()
    # conf_int = lm.conf_int()
    # coefs = pd.DataFrame(
    #     {
    #         "stimulus": motif_names,
    #         "coef": lm.params,
    #         "std err": lm.bse,
    #         "pvalue": smt.multipletests(lm.pvalues, method="sidak")[1],
    #         "coef_lcl": conf_int[0],
    #         "coef_ucl": conf_int[1],
    #     }
    # ).reset_index(drop=True)
    # coefs["responsive"] = (coefs.coef > 0) & (coefs.pvalue < 0.05)
    # if args.output_dir is not None:
    #     outfile = args.output_dir / f"{args.unit}_rate_coefs.csv"
    #     coefs.insert(0, "unit", args.unit)
    #     logging.info("- wrote parameter estimates to %s", outfile)
    # else:
    #     outfile = sys.stdout
    # coefs.to_csv(outfile, index=False)


if __name__ == "__main__":
    main()
