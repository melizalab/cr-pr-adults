#!/usr/bin/env python
# -*- mode: python -*-
""" Estimate population decodability """
import datetime
import json
import logging
import pickle
from pathlib import Path

import ewave
import numpy as np
import pandas as pd
import samplerate
from appdirs import user_cache_dir
from dlab import nbank, pprox
from gammatone.filters import erb_space
from gammatone.gtgram import gtgram, gtgram_strides
from joblib import Memory
from scipy.linalg import hankel
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_cache_dir = user_cache_dir("preconstruct", "melizalab")
_mem = Memory(_cache_dir, verbose=0)

# analysis parameters are hard-coded here
__version__ = "20241003-2"
desired_time_step = 0.0025  # s
desired_sampling_rate = 20000  # Hz
spectrogram_params = {
    "window_time": 0.005,  # s
    "channels": 40,
    "f_min": 1000,  # Hz
    "f_max": 8500,  # Hz
}
spectrogram_compression = 10.0
decoder_window = (0.0, 0.2)  # s
n_basis = 20
linearity_factor = 20
alpha_candidates = np.logspace(-1, 7, 30)
clean_dBFS = -100


class MotifSplitter:
    """Used with pprox.split_trial for splitting long responses into constituent units"""

    def __init__(self, resource_ids):
        self.stim_info = {}
        for result in nbank.describe_many(nbank.default_registry, *resource_ids):
            metadata = result["metadata"]
            metadata["foreground"] = metadata["foreground"].split("-")
            self.stim_info[result["name"]] = pd.DataFrame(metadata)

    def __call__(self, resource_id: str) -> pd.DataFrame:
        return self.stim_info[resource_id]


def compute_spectrogram(row):
    duration = row.samples.size / row.sample_rate
    _, hop_samples, _ = gtgram_strides(
        row.sample_rate,
        spectrogram_params["window_time"],
        desired_time_step,
        row.samples.size,
    )
    hop_time = hop_samples / row.sample_rate
    # this calculation is cached
    spectrogram = _mem.cache(gtgram)(
        row.samples, row.sample_rate, hop_time=desired_time_step, **spectrogram_params
    )
    _, nframes = spectrogram.shape
    spectrogram = np.log10(spectrogram + spectrogram_compression) - np.log10(
        spectrogram_compression
    )
    index = np.arange(0.0, duration, hop_time)[:nframes]
    columns = erb_space(
        spectrogram_params["f_min"],
        spectrogram_params["f_max"],
        spectrogram_params["channels"],
    )[::-1]
    return pd.DataFrame(spectrogram.T, columns=columns, index=index).rename_axis(
        index="time", columns="frequency"
    )


def pool_spikes(x):
    try:
        return np.concatenate(x.dropna().values)
    except ValueError:
        return np.nan


def make_cosine_basis(
    n_tau: int, n_basis: int, linearity_factor: float = 10
) -> np.ndarray:
    """Make a nonlinearly stretched basis consisting of raised cosines

    n_tau:  number of time points
    n_basis:     number of basis vectors
    linearity_vactor:   offset for nonlinear stretching of x axis (larger values -> more linear spacing)
    """
    _min_offset = 1e-20
    first_peak = np.log(linearity_factor + _min_offset)
    last_peak = np.log(n_tau * (1 - 1.5 / n_basis) + linearity_factor + _min_offset)
    peak_centers = np.linspace(first_peak, last_peak, n_basis)
    peak_spacing = (last_peak - first_peak) / (n_basis - 1)
    log_domain = np.log(np.arange(n_tau) + linearity_factor + _min_offset)
    basis = []
    for center in peak_centers:
        cos_input = np.clip(
            (log_domain - center) * np.pi / peak_spacing / 2, -np.pi, np.pi
        )
        cos_basis = (np.cos(cos_input) + 1) / 2
        basis.append(cos_basis / np.linalg.norm(cos_basis))
    return np.column_stack(basis)


def compare_spectrograms_cor(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compare two spectrograms using correlation coefficient across the entire stimulus"""
    cc = np.corrcoef(actual.flat, predicted.flat)
    return cc[0, 1]


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
        default=".",
        help="the directory to store output files",
    )
    parser.add_argument(
        "--pprox-dir",
        type=Path,
        default=None,
        help="search this directory for pprox files before the registry",
    )
    parser.add_argument(
        "--n-units",
        "-n",
        type=int,
        help="fit the decoder to a subset comprising this many units",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1024,
        help="set the seed for the random number generator",
    )
    parser.add_argument(
        "--predict-noisy",
        action="store_true",
        help="if set, compute predictions from responses to noisy stimuli",
    )
    parser.add_argument(
        "units",
        help="path of a file with a list of units, or name of a site to analyze",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        format="%(message)s", level=logging.DEBUG if args.debug else logging.INFO
    )
    logging.info("- start time: %s", datetime.datetime.now())
    logging.info("- version: %s", __version__)

    try:
        unit_file = Path(args.units)
        unit_names = [line.strip() for line in open(unit_file)]
        logging.info(
            "- loading responses for %d units in %s", len(unit_names), unit_file
        )
        dataset_name = unit_file.stem
    except FileNotFoundError:
        logging.info("- loading responses from units in site '%s'", args.units)
        unit_names = [
            record["name"]
            for record in nbank.search(
                nbank.default_registry, name=args.units, dtype="spikes-pprox"
            )
        ]
        logging.info("  - site has %d units", len(unit_names))
        dataset_name = args.units
    logging.info("  - dataset name: %s", dataset_name)

    if args.n_units is not None:
        if args.n_units > len(unit_names):
            parser.error(
                "Number of units in the subsample must be less than the number in the dataset"
            )
        logging.info(
            "  - randomly selecting %d units (seed=%d)", args.n_units, args.random_seed
        )
        rng = np.random.default_rng(args.random_seed)
        unit_names = rng.permutation(unit_names)[: args.n_units]
    else:
        args.n_units = len(unit_names)
        args.random_seed = 0

    all_trials = []
    for unit_name, path in nbank.find_resources(*unit_names, alt_base=args.pprox_dir):
        # this will raise an error if the file was not found
        pprox_data = json.loads(path.read_text())
        # only clean stimuli
        all_trials.extend(
            trial | {"unit": unit_name}
            for trial in pprox_data["pprox"]
            # if trial["stimulus"]["name"].endswith("-100")
        )

    logging.info("  - loaded %d trials", len(all_trials))

    logging.info("- splitting trials into individual motifs")
    long_stim_names = {trial["stimulus"]["name"] for trial in all_trials}
    splitter = MotifSplitter(long_stim_names)
    recording = []
    for trial in all_trials:
        trial_split = pprox.split_trial(trial, splitter)
        trial_split["unit"] = trial["unit"]
        recording.append(trial_split)
    recording = (
        pd.concat(recording)
        .drop(columns=["foreground-dBFS", "background"])
        .rename(columns={"foreground": "stimulus"})
        .set_index(["background-dBFS", "unit", "stimulus"])
        .sort_index()
    )

    logging.info("- loading stimuli")
    stim_names = recording.index.get_level_values("stimulus").unique()
    stimuli = []
    for stim_name, stim_path in nbank.find_resources(*stim_names):
        with ewave.open(stim_path, "r") as fp:
            samples = ewave.rescale(fp.read(), "f")
            resampled = samplerate.resample(
                samples, 1.0 * desired_sampling_rate / fp.sampling_rate, "sinc_best"
            )
            stimuli.append(
                {
                    "stimulus": stim_name,
                    "samples": resampled,
                    "sample_rate": desired_sampling_rate,
                }
            )

    stim_data = pd.DataFrame.from_records(stimuli).set_index("stimulus")
    logging.info("- preprocessing stimuli")
    stims_processed = pd.concat(
        {index: compute_spectrogram(row) for index, row in stim_data.iterrows()},
        names=("stimulus", "time"),
    )

    logging.info("- pooling and binning responses for background-dBFS (%d)", clean_dBFS)

    def bin_responses(trials):
        stim = trials.name
        interval_end = trials.interval_end.iloc[0]
        stim_bins = stims_processed.loc[stim].index.to_numpy()
        time_step = stim_bins[1] - stim_bins[0]
        edges = np.concatenate(
            [
                stim_bins,
                np.arange(stim_bins[-1], interval_end + time_step, time_step)[1:],
            ]
        )
        rates = np.column_stack(
            trials.apply(
                lambda df: np.histogram(df.events, bins=edges)[0] / df.trials, axis=1
            )
        )
        return pd.DataFrame(
            rates,
            index=pd.Index(edges[:-1], name="time"),
            columns=trials.index.get_level_values(0),
        )

    clean_rate_data = (
        recording.loc[clean_dBFS]
        .groupby(["unit", "stimulus"])
        .agg(
            events=pd.NamedAgg(column="events", aggfunc=pool_spikes),
            trials=pd.NamedAgg(column="events", aggfunc=len),
            interval_end=pd.NamedAgg(column="interval_end", aggfunc="max"),
        )
        .groupby("stimulus")
        .apply(bin_responses)
    )

    logging.info("- delay-embedding responses")

    def delay_embed_trial(resp):
        trial = resp.name
        resp = resp.droplevel(0)
        stim_bins = stims_processed.loc[trial].index
        time_step = stim_bins[1] - stim_bins[0]
        lag_range = pd.Index(
            np.arange(decoder_window[0], decoder_window[1], time_step), name="lag"
        )
        # this should be the same for all stims but it's easier to calculate here
        basis_matrix = make_cosine_basis(lag_range.size, n_basis, linearity_factor)

        def delay_embed_unit(unit):
            col = unit.loc[slice(stim_bins[0] - decoder_window[0], stim_bins[-1])]
            row = unit.loc[
                slice(stim_bins[-1], stim_bins[-1] + decoder_window[1])
            ].iloc[: lag_range.size]
            lagged = hankel(col, row)
            return pd.DataFrame(np.dot(lagged, basis_matrix), index=col.index)

        return pd.concat(
            {unit_name: delay_embed_unit(resp[unit_name]) for unit_name in unit_names},
            axis=1,
            names=("unit", "lag"),
        )

    clean_rates_embedded = clean_rate_data.groupby("stimulus").apply(delay_embed_trial)
    # this is really important to ensure that all rows match in the two dataframes
    clean_rates_embedded, clean_stims_processed = clean_rates_embedded.align(
        stims_processed, join="left", axis=0
    )
    assert (
        clean_rates_embedded.shape[0] == clean_stims_processed.shape[0]
    ), "dimensions of data don't match"
    assert all(
        clean_rates_embedded.index == clean_stims_processed.index
    ), "indices of data don't match"

    # won't perfectly align with stimuli but should be close enough
    n_folds = len(stim_names)
    logging.info("- %d-fold cross-validating for hyperparameters", n_folds)
    logging.info(
        "  -  X shape is %s, Y shape is %s",
        clean_rates_embedded.shape,
        clean_stims_processed.shape,
    )
    ridge = Pipeline(
        [("scaler", StandardScaler()), ("ridge", Ridge(fit_intercept=True))]
    )
    xval = GridSearchCV(
        ridge,
        cv=n_folds,
        param_grid={"ridge__alpha": alpha_candidates},
        n_jobs=2,
        verbose=1,
    )
    xval.fit(clean_rates_embedded.values, clean_stims_processed.values)

    best_alpha = xval.best_params_["ridge__alpha"]
    logging.info(
        "  -  best alpha: %.2f; mean score: %.2f", best_alpha, xval.best_score_
    )

    logging.info("- computing predictions for individual motifs")
    correlations = []
    noise_levels = (
        recording.index.get_level_values("background-dBFS").unique().drop(clean_dBFS)
    )

    for motif_name in stim_names:
        X_train = clean_rates_embedded.drop(motif_name)
        Y_train = clean_stims_processed.drop(motif_name)
        X_test = clean_rates_embedded.loc[motif_name]
        Y_test = clean_stims_processed.loc[motif_name]
        ridge = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=best_alpha, fit_intercept=True)),
            ]
        )
        fitted = ridge.fit(X_train.values, Y_train.values)
        pred = fitted.predict(X_test)
        score = fitted.score(X_test, Y_test)
        corr = compare_spectrograms_cor(Y_test.values, pred)
        correlations.append(
            {
                "motif": motif_name,
                "background_dBFS": clean_dBFS,
                "score_actual": score,
                "corr_coef_actual": corr,
                "score_pred_clean": 1.0,
                "corr_coef_pred_clean": 1.0,
            }
        )
        logging.info(
            "  - %s, noise=%.1f dBFS: score=%.3f, corr=%.3f",
            motif_name,
            -100,
            score,
            corr,
        )

        for noise_level in noise_levels:
            noise_recording = recording.loc[noise_level].xs(
                motif_name, level="stimulus", drop_level=False
            )
            # compute rates
            noise_rate_data = (
                noise_recording.groupby(["unit", "stimulus"])
                .agg(
                    events=pd.NamedAgg(column="events", aggfunc=pool_spikes),
                    trials=pd.NamedAgg(column="events", aggfunc=len),
                    interval_end=pd.NamedAgg(column="interval_end", aggfunc="max"),
                )
                .groupby("stimulus")
                .apply(bin_responses)
            )
            # delay embedding
            noise_rates_embedded = noise_rate_data.groupby("stimulus").apply(
                delay_embed_trial
            )
            # ensure rows match and select the test stimulus only
            noise_rates_embedded, noise_stims_processed = noise_rates_embedded.align(
                stims_processed, join="left", axis=0
            )
            assert (
                noise_rates_embedded.shape[0] == noise_stims_processed.shape[0]
            ), "dimensions of data don't match"
            assert all(
                noise_rates_embedded.index == noise_stims_processed.index
            ), "indices of data don't match"
            pred_noisy = fitted.predict(noise_rates_embedded)
            score_actual = fitted.score(noise_rates_embedded, Y_test)
            score_pred = fitted.score(noise_rates_embedded, pred)
            corr_pred = compare_spectrograms_cor(pred, pred_noisy)
            correlations.append(
                {
                    "motif": motif_name,
                    "background_dBFS": noise_level,
                    "score_actual": score_actual,
                    "corr_coef_actual": compare_spectrograms_cor(
                        Y_test.values, pred_noisy
                    ),
                    "score_pred_clean": score_pred,
                    "corr_coef_pred_clean": corr_pred,
                }
            )
            logging.info(
                "  - %s, noise=%.1f dBFS: score=%.3f, corr=%.3f",
                motif_name,
                noise_level,
                score_pred,
                corr_pred,
            )

    model_file = (
        args.output_dir / f"{dataset_name}_n{args.n_units}_s{args.random_seed}_model"
    )
    data = pd.DataFrame(correlations)
    data.insert(0, "seed", args.random_seed)
    data.insert(0, "n_units", args.n_units)
    data.insert(0, "dataset", dataset_name)
    data.to_csv(model_file.with_suffix(".csv"), index=False)
    with open(model_file.with_suffix(".pkl"), "wb") as fp:
        pickle.dump(
            {
                "model": xval,
                "units": clean_rates_embedded.columns,
                "predictions": correlations,
            },
            fp,
        )
    logging.info("- wrote model parameters to %s", model_file)


if __name__ == "__main__":
    main()
