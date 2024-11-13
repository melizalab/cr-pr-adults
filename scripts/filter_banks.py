# -*- mode: python -*-
"""Functions for creating filter banks to be used in the frequency domain

Almost all of this code is translated from the McDermott and Simoncelli MATLAB code
for synthesizing acoustic textures.

All of the frequency bank functions return both positive and negative frequency
coefficients. 

"""
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike


class FilterBank(NamedTuple):
    """A bank of filters defined for a single with a defined sampling rate"""

    filters: np.ndarray
    locations: np.ndarray
    sampling_rate: float


def reflect_filters(filters: ArrayLike, nfft: int) -> np.ndarray:
    """Reflect a bank of filters for positive frequencies around the nyquist frequency."""
    nfreq, nchan = filters.shape
    output = np.zeros((nfft, nchan))
    output[:nfreq] = filters
    if nfft % 2 == 0:
        output[-(nfreq - 2) :] = filters[-2:0:-1]
    else:
        output[-(nfreq - 1) :] = filters[-1:0:-1]
    return output


def erb_freq(n_erb: float) -> float:
    return 24.7 * 9.265 * (np.exp(n_erb / 9.265) - 1)


def freq_erb(freq_Hz: float) -> float:
    return 9.265 * np.log(1 + freq_Hz / (24.7 * 9.265))


def freq_grid(nfft: int, sampling_rate: float) -> np.ndarray:
    """Returns a frequency grid from DC to Nyquist for an FFT of nfft samples"""
    if nfft % 2 == 0:
        nfreqs = int(nfft / 2)
        max_freq = sampling_rate / 2
    else:
        nfreqs = int(nfft - 1) // 2
        max_freq = sampling_rate * (nfft - 1) / 2 / nfft
    return np.linspace(0, max_freq, nfreqs + 1)


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


def hilbert_filter(n_samples: int) -> np.ndarray:
    """Generate a frequency-domain filter to compute the Hilbert transform"""
    h = np.zeros(n_samples, dtype=np.complex128)
    if n_samples % 2 == 0:
        h[0] = h[n_samples // 2] = 1
        h[1 : n_samples // 2] = 2
    else:
        h[0] = 1
        h[1 : (n_samples + 1) // 2] = 2
    return h


def downsample_filter(down: int, beta: float = 8.0) -> np.ndarray:
    """Generate a lowpass FIR filter for use in downsampling"""
    numtaps = 4 * down + 1
    cutoff = 0.5 / down
    firwin = np.sinc(2 * cutoff * (np.arange(numtaps) - (numtaps - 1) / 2))
    firwin = firwin * np.kaiser(numtaps, beta)
    return firwin / np.sum(firwin)


def erb_cosine(
    n_channels: int,
    *,
    nfft: int,
    sampling_rate: float,
    freq_min: float,
    freq_max: float
) -> FilterBank:
    """Creates a bank of cosine-shaped bandpass filters with
    center frequencies equally spaced on an ERB scale.

    Filters overlap by 50%. The filter bank also includes lowpass and highpass
    filters covering the ends of the frequency scale. The squared filters
    collectively sum to 1.0 so that the transform is reversible.

    - n_channels: number of bandpass channels
    - nfft: the size of the FFT transform
    - sampling_rate: the sampling rate of the input signal (in Hz)
    - freq_min: the center frequency of the lowest-frequency bandpass filter
    - freq_max: the center frequency of the highest-frequency bandpass filter

    """
    freqs = freq_grid(nfft, sampling_rate)
    erbs = freq_erb(freqs)
    erb_cutoffs = np.linspace(freq_erb(freq_min), freq_erb(freq_max), n_channels + 2)
    freq_cutoffs = erb_freq(erb_cutoffs)
    cutoff_idx = freqs.searchsorted(freq_cutoffs)
    lower_idx = cutoff_idx[:-2]
    upper_idx = cutoff_idx[2:] - 1
    erb_peaks = erb_cutoffs[1:-1]
    erb_width = (
        erb_cutoffs[2:] - erb_cutoffs[:-2]
    )  # should be same for all peaks with linearly spaced erbs
    erb_filters = np.zeros((freqs.size, n_channels + 2))  # first and last are LP and HP
    for k, (lo, up) in enumerate(zip(lower_idx, upper_idx)):
        erb_filters[lo:up, k + 1] = np.cos(
            (erbs[lo:up] - erb_peaks[k]) / erb_width[k] * np.pi
        )

    lp_upper = cutoff_idx[1]
    hp_lower = cutoff_idx[-2]
    erb_filters[:lp_upper, 0] = np.sqrt(1 - erb_filters[:lp_upper, 1] ** 2)
    erb_filters[hp_lower:, -1] = np.sqrt(1 - erb_filters[hp_lower:, -2] ** 2)
    return FilterBank(
        filters=reflect_filters(erb_filters, nfft),
        locations=erb_freq(erb_peaks),
        sampling_rate=sampling_rate,
    )


def log_constQ_cosine(
    n_channels: int,
    *,
    nfft: int,
    sampling_rate: float,
    freq_min: float,
    freq_max: float,
    q_factor: float
) -> FilterBank:
    """Creates a bank of cosine-shaped bandpass filters with
    center frequencies equally spaced on a log scale with constant Q-factor.

    The cost of being able to separately set the spacing and Q value is that the
    squared filter responses do not sum to 1.

    - n_channels: number of channels
    - nfft: the size of the FFT transform
    - sampling_rate: the sampling rate of the input signal (in Hz)
    - freq_min: the center frequency of the lowest-frequency bandpass filter
    - freq_max: the center frequency of the highest-frequency bandpass filter
    - q_factor: the Q factor (center frequency divided by bandwidth) of the filters

    """
    freqs = freq_grid(nfft, sampling_rate)
    # Center frequencies are evenly spaced on a log scale
    centers = np.logspace(np.log2(freq_min), np.log2(freq_max), num=n_channels, base=2)

    lower = centers * (1 - 1 / q_factor)
    upper = centers * (1 + 1 / q_factor)
    lower_idx = freqs.searchsorted(lower)
    upper_idx = freqs.searchsorted(upper)
    widths = upper - lower
    mod_filters = np.zeros((freqs.size, n_channels))
    for k, (lo, up) in enumerate(zip(lower_idx, upper_idx)):
        mod_filters[lo:up, k] = np.cos((freqs[lo:up] - centers[k]) / widths[k] * np.pi)

    # not sure what this normalization is exactly; McDermott notes that
    # these filters can't sum to 1
    total_power = np.sum(mod_filters**2, 1)
    norm = np.sqrt(total_power[(freqs >= centers[3]) & (freqs <= centers[-4])].mean())
    return FilterBank(
        filters=reflect_filters(mod_filters / norm, nfft),
        locations=centers,
        sampling_rate=sampling_rate,
    )


def lowpass_autocorr(lags: ArrayLike, *, nfft: int, sampling_rate: float) -> FilterBank:
    """Creates a bank of lowpass filters with cutoffs at defined lags

    - lags: an array of lags in units of samples
    - nfft: the size of the FFT transform
    - sampling_rate: the sampling rate of the input signal (in Hz)

    In the returned FilterBank, the locations are the lags associated with each channel
    """
    lags = np.asarray(lags)
    n_eac_channels = lags.size - 1
    freqs = freq_grid(nfft, sampling_rate)
    eac_filters = np.zeros((freqs.size, n_eac_channels))

    upper = 1 / (4 * (lags[1:] - lags[:-1]) / 1000)
    lower = 0.5 * upper
    lower_idx = freqs.searchsorted(lower, side="right")
    upper_idx = freqs.searchsorted(upper) - 1
    for k, (lo, up) in enumerate(zip(lower_idx, upper_idx)):
        if upper[k] > sampling_rate / 2:
            eac_filters[:, k] = 1
        elif lo < up:
            eac_filters[:lo, k] = 1
            eac_filters[lo:up, k] = np.cos(
                (freqs[lo:up] - freqs[lo]) / (freqs[lo] - freqs[up]) * np.pi / 2
            )
        else:
            eac_filters[:lo, k] = 1

    return FilterBank(
        filters=reflect_filters(eac_filters, nfft),
        locations=lags[1:],
        sampling_rate=sampling_rate,
    )


def octave_cosine(
    *, nfft: int, sampling_rate: float, freq_min: float, freq_max: float
) -> FilterBank:
    """Creates a bank of cosine-shaped bandpass filters with center frequencies
    spaced as octaves from freq_min to freq_max.

    Adjacent filters overlap by 50%. The squared responses of all the filters
    sum to 1 between freq_min and freq_max.

    - nfft: the size of the FFT transform
    - sampling_rate: the sampling rate of the input signal (in Hz)
    - freq_min: the center frequency of the lowest-frequency bandpass filter
    - freq_max: the center frequency of the highest-frequency bandpass filter

    """
    freqs = freq_grid(nfft, sampling_rate)
    if freq_max > sampling_rate / 2:
        freq_max = freqs[-1]

    cutoffs = freq_max / np.power(2, np.arange(21))
    centers = cutoffs[cutoffs > freq_min][:0:-1]  # drop last and reverse

    lower = centers / 2
    upper = centers * 2
    lower_idx = freqs.searchsorted(lower)
    upper_idx = freqs.searchsorted(upper)
    avg = (np.log2(lower) + np.log2(upper)) / 2
    width = np.log2(upper) - np.log2(lower)

    n_oct_channels = centers.size
    oct_filts = np.zeros((freqs.size, n_oct_channels))
    for k, (lo, up) in enumerate(zip(lower_idx, upper_idx)):
        oct_filts[lo:up, k] = np.cos(
            (np.log2(freqs[lo:up]) - avg[k]) / width[k] * np.pi
        )

    return FilterBank(
        filters=reflect_filters(oct_filts, nfft),
        locations=centers,
        sampling_rate=sampling_rate,
    )
