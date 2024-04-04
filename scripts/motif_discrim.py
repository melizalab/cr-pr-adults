#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" Compute discriminability using a classifier between responses to different motifs. """
import json
import logging
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pyspike
from core import (
    MotifSplitter,
    pairwise_spike_comparison,
    split_trials,
    trial_to_spike_train,
    inv_spike_sync_matrix,
)
from dlab import nbank
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score

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
    parser.add_argument("unit", help="name of the unit to analyze")
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

    pprox_file = nbank.find_resource(args.unit)
    logging.info("- reading spikes from %s", pprox_file)
    unit = json.loads(pprox_file.read_text())
    logging.info("- splitting trials by motif")
    splitter = MotifSplitter()
    motifs = split_trials(splitter, unit)

    logging.info(
        "- computing classifier performance using pairwise spike train distances at each background level"
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

    results = spike_trains.groupby("background_dBFS").apply(bootstrap_classifier)

    logging.info(
        "- predicting stimuli at each background level using classifier trained on lowest noise condition"
    )
    classifier = KNeighborsClassifier(
        n_neighbors=args.classifier_neighbors, metric="precomputed"
    )
    st_train = spike_trains.loc[clean_dBFS]
    train_dist = pairwise_spike_comparison(
        st_train, comparison_fun=inv_spike_sync_matrix, stack=False
    )
    group_idx, names = train_dist.index.factorize()
    classifier.fit(train_dist.values, group_idx)

    # for each trial not in the training set, compute spike distances to
    # training trials, then use the classifier to predict which stimulus was
    # presented
    def compare_to_training(st_test):
        return st_train.apply(
            lambda st_ref: 1 - pyspike.spike_sync(st_ref, st_test.spikes)
        ).rename_axis("ref")

    dist_to_clean = (
        spike_trains.drop(clean_dBFS)
        .to_frame("spikes")
        .apply(compare_to_training, axis=1)
    )
    predicted = pd.Series(
        names[classifier.predict(dist_to_clean.values)], index=dist_to_clean.index
    ).rename("predicted")
    accuracy = (
        pd.Series(
            1.0 * (predicted.index.get_level_values(-1) == predicted),
            index=predicted.index,
        )
        .groupby(["background_dBFS", "foreground"])
        .mean()
    )

    results = results.join(accuracy.rename("pred_score")).reset_index()
    if args.output_dir is not None:
        results.insert(0, "unit", args.unit)
        out_path = args.output_dir / f"{args.unit}_motif_discrim.csv"
        logging.info(" - wrote results to %s", out_path)
    else:
        out_path = sys.stdout
    results.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
