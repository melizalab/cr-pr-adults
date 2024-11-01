# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Computes sound texture statistics from acoustic recordings.

This script uses the algorithm developed by McDermott and Simoncelli (2011) to
measure statistics based on a model of the auditory periphery. It takes in a
list of samples and generates an HDF5 file containing the statistics of the
samples, one entry per sample.

"""

import os
import csv
import datetime
import argparse
import logging
from pathlib import Path

import arf
import numpy as np
from scipy import signal
import h5py

import filter_banks as fbs
import core

log = logging.getLogger()  # root logger
__version__ = "20241101-1"

# parameters: pre-processing
desired_duration = 10.0
desired_sampling_rate = 20000
desired_rms = 0.01

# acoustic spectrum parameters
n_channels = 30  # not including lowpass and highpass
freq_min = 200  # center frequency of lowest channel (not including LP)
freq_max = 10000  # center frequency of highest channel (not including HP)

# envelope parameters
envelope_sampling_rate = 400.0
envelope_compression_exponent = 0.3

# modulation spectrum parameters
n_mod_channels = 20
mod_min = 0.5
mod_max = 200
mod_q_value = 2
mod_min_C12 = 1  # this is the lowest frequency in the octave-spaced modulation filterbank used for the C1 and C2 correlations

# histogram parameters
n_hist_bins = 128


def envelope_window(n_frames: int, sampling_rate: float, ramp_len: float) -> np.ndarray:
    """Create a window for tapering envelopes to avoid edge effects.

    The ramps are formed using a squared sine wave. McDermott uses a
    cosine-shaped ramp, which has a somewhat more abrupt takeoff. The window is
    normalized to have a sum of 1.0

    n_frames: the size of the window, in samples
    sampling_rate: the sampling rate of the envelope (in Hz)
    ramp_len: the length of the ramps, in seconds

    """
    env_window = np.ones(n_frames)
    n_ramp = int(sampling_rate * ramp_len)
    ramp = np.sin(np.linspace(0, np.pi / 2, n_ramp)) ** 2
    env_window[:n_ramp] = ramp
    env_window[-n_ramp:] = ramp[::-1]
    return env_window / env_window.sum()


def weighted_moments(x: np.ndarray, window: np.ndarray, axis: int = 0):
    """Compute the weighted first four central moments of x along an axis.

    The second moment is not the variance, but the standard deviation
    divided by the mean. According to McDermott, this is to make it
    unitless like the skew and kurtosis.
    """
    m1 = np.sum(window[:, np.newaxis] * x, axis=axis)
    m2 = np.sum(window[:, np.newaxis] * ((x - m1) ** 2), axis=axis)
    m3 = np.sum(window[:, np.newaxis] * ((x - m1) ** 3), axis=axis)
    m4 = np.sum(window[:, np.newaxis] * ((x - m1) ** 4), axis=axis)
    return [m1, np.sqrt(m2) / m1, m3 / (m2 ** (3 / 2)), m4 / (m2**2)]


def roll_pad(x: np.ndarray, shift: int) -> np.ndarray:
    """Shift a 2D array by shift samples along the first axis, padding with zeros"""
    xr = np.roll(x, shift, axis=0)
    if shift > 0:
        xr[:shift] = 0
    elif shift < 0:
        xr[shift:] = 0
    return xr


def freq_subbands(signal: np.ndarray, filterbank: fbs.FilterBank) -> np.ndarray:
    """Compute the frequency subbands of a 1-D signal (n_samples). Returns
    (n_samples x n_filters) complex array (frequency-domain)

    """
    assert signal.ndim == 1
    assert signal.size == filterbank.filters.shape[0]

    fft_signal = np.fft.fft(signal)
    fft_subbands = filterbank.filters * fft_signal[:, np.newaxis]
    # return np.fft.ifft(fft_subbands, axis=0).real
    return fft_subbands


def subband_envelopes(
    subbands_fft: np.ndarray, *, downsample: int, compression: float = 0.3
) -> np.ndarray:
    """Compute the envelopes of subbands using Hilbert transform.

    - subbands_fft: FFT-transformed subbands (n_samples x n_channels)
    - downsample: factor by which to downsample envelopes
    - compression: exponent for compressing the envelope
    """
    from scipy.signal import resample_poly

    # We generate the subbands by filtering in the frequency domain, so we can
    # save a step in the Hilbert transform by working from this directly.
    nfft, nchannels = subbands_fft.shape
    h = np.zeros(nfft, dtype=subbands_fft.dtype)
    if nfft % 2 == 0:
        h[0] = h[nfft // 2] = 1
        h[1 : nfft // 2] = 2
    else:
        h[0] = 1
        h[1 : (nfft + 1) // 2] = 2
    subbands_hilb = np.fft.ifft(subbands_fft * h[:, np.newaxis], axis=0)
    subband_env = np.power(np.abs(subbands_hilb), compression)
    return np.fmax(resample_poly(subband_env, 1, downsample, axis=0), 0)


def channel_histograms(subbands: np.ndarray, n_bins: int) -> np.ndarray:
    """Compute histograms of values in each subband.

    Histograms are normalized to sum to 1. Bins are set separately for each
    channel. This function can be used with the subbands themselves or the
    envelopes.

    Returns (histograms, bins)

    """
    _, n_channels = subbands.shape
    bins = np.zeros((n_bins + 1, n_channels))
    hist = np.zeros((n_bins, n_channels))
    for k in range(n_channels):
        hist, bins = np.histogram(subbands[:, k], n_bins)
        hist[:, k] = hist
        bins[:, k] = bins
    hist /= hist.sum(axis=0)
    return hist, bins


def corrcoef_windowed(x: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Compute a windowed cross-correlation between channels of a 2-D array

    - x: 2D array (n_frames x n_channels)
    - window: 1D array (n_frames)

    """
    centered = x - x.mean(axis=0)
    stds = np.sqrt(np.mean(centered**2, axis=0))
    return np.dot((window[:, np.newaxis] * centered).T, centered) / np.outer(stds, stds)


def envelope_autocorrelation(
    env_fft: np.ndarray, filterbank: fbs.FilterBank, window: np.ndarray
) -> np.ndarray:
    """Compute the autocorrelation of the envelope in each channel.

    - env_fft: FFT-transform of envelope (n_frames x n_channels)
    - filterbank: autocorrelation filters (n_frames x n_lags).
      Uses the locations attribute for the lags.
    - window: window to remove edge effects (n_frames)

    Returns envelope autocorrelations (n_channels x n_lags).

    """
    n_frames, n_channels = env_fft.shape
    _n_frames, n_lags = filterbank.filters.shape
    assert n_frames == _n_frames

    acorr_env = np.zeros((n_channels, n_lags))
    for k, n_samples in enumerate(filterbank.locations):
        fft_env_filt = filterbank.filters[:, k][:, np.newaxis] * env_fft
        env_filt = np.fft.ifft(fft_env_filt, axis=0).real
        env_filt -= env_filt.mean(axis=0)
        env_filt_var = (env_filt**2).mean(0)
        acorr_env[:, k] = (
            np.sum(
                window[:, np.newaxis]
                * roll_pad(env_filt, -n_samples)
                * roll_pad(env_filt, n_samples),
                0,
            )
            / env_filt_var
        )
    return acorr_env


def envelope_modulation_power(
    env_fft: np.ndarray, filterbank: fbs.FilterBank, window: np.ndarray
) -> np.ndarray:
    """Compute the modulation power spectrum of the envelope in each channel.

    The modulation bands are calculated using constant-Q log-spaced filters (see
    filter_banks.log_constQ_cosine).

    - env_fft: FFT transform of envelope (n_frames x n_channels)
    - filterbank: modulation power filters (n_frames x n_mod_freqs)
    - window: window to remove edge effects (n_frames)

    Returns the un-normalized modulation power spectra (n_channels x
    n_mod_freqs). Divide each channel by its variance to get the normalized MPS
    used in texture synth.

    """
    n_frames, n_channels = env_fft.shape
    _n_frames, n_mod_freqs = filterbank.filters.shape
    assert n_frames == _n_frames

    mod_env = np.zeros((n_channels, n_mod_freqs))
    for k in range(n_mod_freqs):
        fft_env_filt = filterbank.filters[:, k][:, np.newaxis] * env_fft
        env_filt = np.fft.ifft(fft_env_filt, axis=0).real
        mod_env[:, k] = np.sum(window[:, np.newaxis] * env_filt**2, axis=0)
    return mod_env


def envelope_c1_correlations(
    env_fft: np.ndarray, filterbank: fbs.FilterBank, window: np.ndarray
) -> np.ndarray:
    """Compute the cross-acoustic-band correlations in each modulation band.

    The modulation bands are calculated using octave-spaced filters (see
    filter_banks.octave_cosine). The first filter is not used.

    - env_fft: FFT transform of envelope (n_frames x n_channels)
    - filterbank: modulation power filters (n_frames x n_mod_freqs)
    - window: window to remove edge effects (n_frames)


    Returns (n_channels x n_channels x n_mod_freqs - 1) array.
    """
    n_frames, n_channels = env_fft.shape
    _n_frames, n_oct_freqs = filterbank.filters.shape
    assert n_frames == _n_frames

    corr = np.zeros((n_channels, n_channels, n_oct_freqs - 1))
    # first octave-spaced filter is not used
    for k in range(1, n_oct_freqs):
        fft_env_filt = filterbank.filters[:, k][:, np.newaxis] * env_fft
        env_filt = np.fft.ifft(fft_env_filt, axis=0).real
        corr[:, :, k - 1] = corrcoef_windowed(env_filt, window)
    return corr


def envelope_c2_correlations(
    env_fft: np.ndarray, filterbank: fbs.FilterBank, window: np.ndarray
) -> np.ndarray:
    """Compute the cross-modulation-band correlations in each acoustic band.

    The modulation bands are calculated using octave-spaced filters (see
    filter_banks.octave_cosine).

    - env_fft: FFT transform of envelope (n_frames x n_channels)
    - filterbank: modulation power filters (n_frames x n_mod_freqs)
    - window: window to remove edge effects (n_frames)

    Returns (n_channels x n_mod_freqs x 2) array.

    """
    n_frames, n_channels = env_fft.shape
    _n_frames, n_oct_freqs = filterbank.filters.shape
    assert n_frames == _n_frames

    corr = np.zeros((n_channels, n_oct_freqs - 1, 2))
    # pregenerate the unit step function for calculating hilbert transform
    h = np.zeros(n_frames, dtype=env_fft.dtype)
    if n_frames % 2 == 0:
        h[0] = h[n_frames // 2] = 1
        h[1 : n_frames // 2] = 2
    else:
        h[0] = 1
        h[1 : (n_frames + 1) // 2] = 2

    for k in range(n_channels):
        fft_env_filt = filterbank.filters * env_fft[:, k][:, np.newaxis]
        # env_filt = np.fft.ifft(fft_env_filt, axis=0).real
        # analytic = signal.hilbert(env_filt, axis=0)
        analytic = np.fft.ifft(fft_env_filt * h[:, np.newaxis], axis=0)
        coarse = np.real((analytic**2) / np.abs(analytic))[:, :-1]
        fw_analytic_real = np.real(analytic[:, 1:])
        fw_analytic_imag = np.imag(analytic[:, 1:])
        sig_cw = np.sqrt(np.sum(window[:, np.newaxis] * coarse**2, axis=0))
        sig_fw = np.sqrt(np.sum(window[:, np.newaxis] * fw_analytic_real**2, axis=0))
        corr[k, :, 0] = np.sum(
            window[:, np.newaxis] * coarse * fw_analytic_real, axis=0
        ) / (sig_cw * sig_fw)
        corr[k, :, 1] = np.sum(
            window[:, np.newaxis] * coarse * fw_analytic_imag, axis=0
        ) / (sig_cw * sig_fw)
    return corr


def write_filterbank(group: h5py.Group, name: str, filterbank: fbs.FilterBank):
    dset = group.create_dataset(name, data=filterbank.filters)
    # dset_loc = group.create_dataset(f"{name}_locations", data=filterbank.locations)
    dset.attrs["locations"] = filterbank.locations
    dset.attrs["sampling_rate"] = filterbank.sampling_rate
    return dset


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Calculate sound texture statistics of recording(s)"
    )
    p.add_argument(
        "-v", "--version", action="version", version="%(prog)s " + __version__
    )
    p.add_argument("--debug", help="show verbose log messages", action="store_true")
    p.add_argument(
        "--scaling",
        type=float,
        help="if set, will apply this scaling factor to all recordings instead of normalizing",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=desired_duration,
        help="recordings are truncated to this duration (in s) and skipped if they are shorter (default %(default)s s)",
    )
    p.add_argument(
        "--output",
        type=Path,
        help="hdf5 file for output (overwrites existing files)",
    )
    p.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="wave files to analyze",
    )
    args = p.parse_args()
    core.setup_log(log, args.debug)

    if args.output:
        log.info("- opening %s for output", args.output)
        outfile = h5py.File(args.output, "w")
    else:
        log.info("- no output (dry run)")
        outfile = h5py.File("dummy", "w", driver="core", backing_store=False)

    # initialize the filterbanks
    ds_factor = desired_sampling_rate / envelope_sampling_rate
    n_samples = args.duration * desired_sampling_rate
    n_samples = int(np.floor(n_samples / ds_factor / 2) * ds_factor * 2)
    desired_duration = n_samples / desired_sampling_rate
    n_frames = int(n_samples / ds_factor)

    log.info(
        "- generating filterbanks for %.1f s input (n_samples: %d, n_frames: %d)",
        desired_duration,
        n_samples,
        n_frames,
    )
    env_window = envelope_window(n_frames, envelope_sampling_rate, 1.0)
    erb_filters = fbs.erb_cosine(
        n_channels,
        nfft=n_samples,
        sampling_rate=desired_sampling_rate,
        freq_min=freq_min,
        freq_max=freq_max,
    )
    mps_filters = fbs.log_constQ_cosine(
        n_mod_channels,
        nfft=n_frames,
        sampling_rate=envelope_sampling_rate,
        freq_min=mod_min,
        freq_max=mod_max,
        q_factor=mod_q_value,
    )
    c12_filters = fbs.octave_cosine(
        nfft=n_frames,
        sampling_rate=envelope_sampling_rate,
        freq_min=mod_min_C12,
        freq_max=mod_max,
    )
    # if args.save_filters:
    write_filterbank(outfile, "erb_filters", erb_filters)
    write_filterbank(outfile, "mps_filters", mps_filters)
    write_filterbank(outfile, "c12_filters", c12_filters)

    for i, wavfile in enumerate(args.inputs):
        log.info("- %s:", wavfile)
        sample = core.load_wave(wavfile)
        mtime = wavfile.stat().st_mtime
        timestamp = datetime.datetime.fromtimestamp(mtime)
        log.info("  - preprocessing recording")
        core.resample(sample, desired_sampling_rate)
        try:
            core.truncate(sample, n_samples)
        except ValueError as err:
            log.info("skipping %s: %s", sample["name"], err)
            continue
        if args.scaling is None:
            log.info("    - normalizing RMS amplitude to %.2f", desired_rms)
            core.rescale(sample, core.dBFS(desired_rms))
        else:
            log.info("    - applying scaling factor of %.0f", args.scaling)
            sample["signal"] *= args.scaling
            sample["dBFS"] = core.dBFS(sample["signal"])

        out_group = arf.create_entry(
            outfile,
            f"sample_{i:04}",
            timestamp,
            source_file=wavfile.name,
        )

        log.info("  - computing subband envelopes")
        subbands_fft = freq_subbands(sample["signal"], erb_filters)
        envelopes = subband_envelopes(
            subbands_fft,
            downsample=ds_factor,
            compression=envelope_compression_exponent,
        )
        fft_env = np.fft.fft(envelopes, axis=0)
        out_group.create_dataset("envelopes", data=envelopes)

        log.info("  - computing envelope statistics")
        env_mean, env_std, env_skew, env_kurt = weighted_moments(
            envelopes, env_window, axis=0
        )
        out_group.create_dataset("envelope_means", data=env_mean)
        out_group.create_dataset("envelope_stdvs", data=env_std)
        out_group.create_dataset("envelope_skew", data=env_skew)
        out_group.create_dataset("envelope_kurtosis", data=env_kurt)
        env_var = np.sum(
            env_window[:, np.newaxis] * (envelopes - env_mean) ** 2, axis=0
        )
        env_mps = (
            envelope_modulation_power(fft_env, mps_filters, env_window)
            / env_var[:, np.newaxis]
        )
        out_group.create_dataset("modulation_power", data=env_mps)
        env_corr = corrcoef_windowed(envelopes, env_window)
        out_group.create_dataset("envelope_corr", data=env_corr)
        env_c1 = envelope_c1_correlations(fft_env, c12_filters, env_window)
        out_group.create_dataset("envelope_c1_correlations", data=env_c1)
        env_c2 = envelope_c2_correlations(fft_env, c12_filters, env_window)
        out_group.create_dataset("envelope_c2_correlations", data=env_c2)

    outfile.close()
