#!/usr/bin/env python
# -*- mode: python -*-
""" Compute discriminability using a classifier between responses to different motifs. """
import json
import logging
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from core import (
    MotifSplitter,
    inv_spike_sync_matrix,
    pairwise_spike_comparison,
    split_trials,
    trial_to_spike_train,
)
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# For discriminability analysis, we clip all responses to the shortest stimulus
# to remove information encoded in the response length. This stimulus is much
# shorter than the others, so it's excluded to avoid having to clip too much.
stimuli_to_drop = ["igmi8fxa"]
clean_dBFS = -100


class ShuffledLeaveOneOut(LeaveOneOut):
    """This scikit-learn splitter shuffles the order of the training set; otherwise if there
    are a lot of ties in the distance matrix, the classifier can end up always
    picking the same group. This becomes a problem if you want to interpret the
    performance of the classifier on individual stimuli.

    """

    def __init__(self, rng):
        super().__init__()
        self.rng = rng

    def split(self, *args, **kwargs):
        for train, test in super().split(*args, **kwargs):
            yield self.rng.permutation(train), test


def kneighbors_classifier(distance_matrix, rng, n_neighbors=9):
    """Compute cross-validated performance of a k-neighbors classifier on the spike distance matrix"""
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, metric="precomputed")
    loo = ShuffledLeaveOneOut(rng)
    groups = distance_matrix.index
    group_idx, _ = pd.factorize(groups)
    cv_results = cross_val_score(neigh, distance_matrix.values, group_idx, cv=loo)
    return pd.Series(cv_results, index=groups).groupby(level=0).mean().rename("score")


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug", help="show verbose log messages", action="store_true"
    )
    parser.add_argument(
        "--shuffle-replicates",
        type=int,
        default=100,
        help="number of times to compute distances with shuffled responses",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=1204,
        help="seed for random number generator used in bootstrapping",
    )
    parser.add_argument(
        "--classifier-neighbors",
        "-k",
        type=int,
        default=9,
        help="number of neighbors (k parameter) in classifier",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help="the directory to store output files (if not set, prints to stdout)",
    )
    parser.add_argument(
        "--metadata-dir",
        "-m",
        type=Path,
        required=True,
        help="the directory where stimulus metadata files are stored",
    )
    parser.add_argument("unit", type=Path, help="pprox data (one unit) to analyze")
    args = parser.parse_args(argv)
    logging.basicConfig(
        format="%(message)s", level=logging.DEBUG if args.debug else logging.INFO
    )
    logging.info("- setting random seed: %d", args.bootstrap_seed)
    rng = np.random.default_rng(args.bootstrap_seed)
    logging.info(
        "- bootstrap will use %d replicates for z-scores",
        args.shuffle_replicates,
    )
    logging.info("- excluding these motifs: %s", ", ".join(stimuli_to_drop))

    pprox_file = args.unit
    logging.info("- reading spikes from %s", pprox_file)
    unit = json.loads(pprox_file.read_text())
    logging.info("- splitting trials by motif")
    splitter = MotifSplitter()
    motifs = split_trials(splitter, unit, args.metadata_dir)

    logging.info(
        "- computing classifier performance using pairwise spike train distances"
    )
    trials = motifs.drop(stimuli_to_drop, level="foreground").rename_axis(
        index=lambda s: s.replace("-", "_")
    )
    spike_trains = trials.apply(
        partial(trial_to_spike_train, interval_end=trials.interval_end.min()), axis=1
    )

    def bootstrap_classifier(trials):
        rates = (
            trials.apply(lambda trial: trial.spikes.size)
            .groupby("foreground")
            .agg(["mean", np.count_nonzero])
            .rename(columns={"mean": "spikes_mean", "count_nonzero": "nonzero_trials"})
            .rename_axis(index="ref")
        )
        distances = pairwise_spike_comparison(
            trials,
            shuffle=False,
            add_spikes=0,
            rng=rng,
            comparison_fun=inv_spike_sync_matrix,
            stack=False,
        )
        scores = kneighbors_classifier(
            distances, rng=rng, n_neighbors=args.classifier_neighbors
        )
        shuffled = pd.concat(
            [
                pairwise_spike_comparison(
                    trials,
                    shuffle=True,
                    add_spikes=0,
                    rng=rng,
                    comparison_fun=inv_spike_sync_matrix,
                    stack=False,
                ).pipe(
                    kneighbors_classifier,
                    rng=rng,
                    n_neighbors=args.classifier_neighbors,
                )
                for i in range(args.shuffle_replicates)
            ],
            keys=range(args.shuffle_replicates),
            names=["replica"],
        )
        # the individual motifs are z-scored based on average and std across all
        # replicates and motifs
        shuffled_mean = shuffled.mean()
        shuffled_std = shuffled.std()
        z_scores = (scores.rename("z_score") - shuffled_mean) / shuffled_std
        # to z-score the average performance, we use the standard deviation of
        # the average
        score_avg = scores.mean()
        scores.loc["_average"] = score_avg
        z_scores.loc["_average"] = (score_avg - shuffled_mean) / shuffled.groupby(
            "replica"
        ).mean().std()
        rates.loc["_average"] = rates.mean()
        return (
            pd.merge(scores, z_scores, left_index=True, right_index=True)
            .join(rates)
            .rename_axis(index="foreground")
        )

    # for this study we only use the responses in the 'clean' condition
    results = spike_trains.loc[clean_dBFS].pipe(bootstrap_classifier).reset_index()

    if args.output_dir is not None:
        unit_name = args.unit.stem
        out_path = args.output_dir / f"{unit_name}_motif_discrim.csv"
        results.insert(0, "unit", unit_name)
        logging.info(" - wrote results to %s", out_path)
    else:
        out_path = sys.stdout
    results.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
