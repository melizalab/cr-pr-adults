# -*- mode: python -*-
""" Core functions for processing responses and stimuli """
import logging
from typing import Callable, Dict, TypedDict

import numpy as np
import pandas as pd
import pyspike
from dlab import nbank, pprox


def setup_log(log, debug=False):
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    loglevel = logging.DEBUG if debug else logging.INFO
    log.setLevel(loglevel)
    ch.setLevel(loglevel)
    ch.setFormatter(formatter)
    log.addHandler(ch)


class Waveform(TypedDict):
    name: str
    signal: np.ndarray
    sampling_rate: float
    duration: float
    dBFS: float


class NullSplitter:
    """This 'splitter' returns a single row that annotates each trial with metadata"""

    def __call__(self, resource_id: str, stim_info: Dict[str, Dict]) -> pd.DataFrame:
        metadata = stim_info[resource_id]
        metadata["stim_begin"] = [0.0]
        metadata["stim_end"] = [metadata.pop("background-duration")]
        return pd.DataFrame(metadata)


class ForegroundSplitter:
    """Split out the pre-foreground, foreground, and post-foreground sections from auditory scenes

    For brevity, the foreground is named after the first motif. The
    post-foreground interval is taken to be as long as the pre-foreground
    interval.

    """

    def __call__(self, resource_id: str, stim_info: Dict[str, Dict]) -> pd.DataFrame:
        metadata = stim_info[resource_id]
        pre_padding = metadata["stim_begin"][0]
        metadata["foreground"] = [
            "before",
            metadata["foreground"].split("-")[0],
            "after",
        ]
        metadata["stim_begin"] = [
            0,
            metadata["stim_begin"][0],
            metadata["stim_end"][-1],
        ]
        metadata["stim_end"] = [
            pre_padding,
            metadata["stim_end"][-1],
            metadata["stim_end"][-1] + pre_padding,
        ]
        return pd.DataFrame(metadata)


class MotifSplitter:
    """Look up splits for motifs in stimulus."""

    def __call__(self, resource_id: str, stim_info: Dict[str, Dict]) -> pd.DataFrame:
        metadata = stim_info[resource_id]
        metadata["foreground"] = metadata["foreground"].split("-")
        return pd.DataFrame(metadata)


class MotifBackgroundSplitter(MotifSplitter):
    """This is like MotifSplitter but it includes splits for the silence
    before the stimulus and the segment of background before the first motif.

    """

    def __init__(self, silence_start=-1.0):
        self.silence_start = silence_start

    def __call__(self, resource_id: str, stim_info: Dict[str, Dict]) -> pd.DataFrame:
        metadata = stim_info[resource_id]
        metadata["foreground"] = metadata["foreground"].split("-")
        metadata["foreground"].insert(0, "background")
        metadata["stim_end"].insert(0, metadata["stim_begin"][0])
        metadata["stim_begin"].insert(0, 0.0)
        metadata["foreground"].insert(0, "silence")
        metadata["stim_end"].insert(0, 0.0)
        metadata["stim_begin"].insert(0, self.silence_start)
        return pd.DataFrame(metadata)


def split_trials(
    splitter: Callable[[str, Dict], pd.DataFrame], trials: pprox.Collection
) -> pd.DataFrame:
    """For each trial, split into motifs using splitter"""
    # this is basically a wrapper around pprox.split_trial that caches
    stim_names = [trial["stimulus"]["name"] for trial in trials["pprox"]]
    stim_info = {
        result["name"]: result["metadata"]
        for result in nbank.describe_many(nbank.default_registry, *stim_names)
    }

    def wrapper(resource_id):
        return splitter(resource_id, stim_info)

    return (
        pd.concat([pprox.split_trial(trial, wrapper) for trial in trials["pprox"]])
        .set_index(["background-dBFS", "foreground"])
        .sort_index()
    )


def trial_to_spike_train(trial, interval_end=None):
    """convert trial record into a pyspike.Spiketrain object

    - empty trials are converted from np.nan to empty list
    - duplicate events are dropped
    - events after interval_end are dropped
    """
    from numpy import unique

    if isinstance(trial.events, float):
        events = []
    elif interval_end is not None:
        events = [e for e in trial.events if e < interval_end]
    else:
        events = trial.events
    if interval_end is None:
        interval_end = trial.interval_end
    return pyspike.SpikeTrain(unique(events), interval_end)


def add_random_spikes(spike_train, rng, n=1):
    """If spike_train is empty, add n random spikes between the start and end"""
    if len(spike_train) > 0:
        return spike_train
    new_spikes = rng.uniform(low=spike_train.t_start, high=spike_train.t_end, size=n)
    sorted_spikes = sorted(spike_train.spikes.tolist() + new_spikes.tolist())
    return pyspike.SpikeTrain(sorted_spikes, spike_train.t_end)


def pairwise_spike_comparison(
    spike_trains,
    shuffle=False,
    add_spikes=0,
    rng=None,
    comparison_fun=pyspike.spike_distance_matrix,
    stack=True,
):
    """Compute a spike distance/similarity between all pairs of spike trains.

    This function takes in a pandas series of spike trains. The last level of
    the series index should contain the cluster label. It will return a pandas
    series where the last two levels indicate the reference and target cluster
    labels.

    If shuffle=True, the spike trains are shuffled prior to computing
    the distance matrix.

    If add_spikes is greater than zero, this number of spikes is randomly added
    to each empty trial. This is useful in regularizing measures for
    very selective neurons (because the distance between two empty spike trains
    is otherwise zero).

    rng must be set to a random number generator (e.g. np.random.default_rng())
    if shuffling or adding spikes.

    """
    names = spike_trains.index.get_level_values(level=-1)
    if add_spikes > 0:
        spike_trains = spike_trains.apply(add_random_spikes, args=(rng, add_spikes))
    if shuffle:
        spike_trains = rng.permutation(spike_trains)
    else:
        spike_trains = spike_trains.to_list()
    df = pd.DataFrame(
        comparison_fun(spike_trains),
        index=names.rename("ref"),
        columns=names.rename("tgt"),
    )
    if stack:
        return df.stack().sort_index()
    else:
        return df


def inv_spike_sync_matrix(*args, **kwargs):
    return 1 - pyspike.spike_sync_matrix(*args, **kwargs)


def df_extent(df):
    """Returns the extent (x0, x1, y0, y1) of a pandas dataframe"""
    from numpy import arange

    index = df.index
    if not pd.api.types.is_numeric_dtype(index):
        index = arange(index.size)
    columns = df.columns
    if not pd.api.types.is_numeric_dtype(columns):
        columns = arange(columns.size)
    return (columns[0], columns[-1], index[0], index[-1])


class ZCA:
    """Zero-phase whitening (could be used instead of PCA)"""

    def fit_transform(self, X):
        n = X.shape[0]
        self.x_mean = X.mean(0)
        X_centered = X - self.x_mean
        cov = np.dot(X_centered.T, X_centered) / n
        V, S, Vt = np.linalg.svd(cov)
        self.whiten_ = np.dot(V / np.sqrt(S + 1e-10), Vt)
        self.color_ = np.dot(np.sqrt(S + 1e-10) * V, Vt)
        return np.dot(X_centered, self.whiten_)

    def transform(self, X):
        return np.dot(X - self.x_mean, self.whiten_)

    def inverse_transform(self, U):
        # technically the transpose but the matrix is symmetric
        return np.dot(U, self.color_) + self.x_mean
