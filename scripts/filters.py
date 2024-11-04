# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Functions to perform filtering in the time domain

This code is mostly from https://github.com/endolith/waveform_analysis
"""
from enum import Enum
from typing import Tuple

import numpy as np
from scipy.signal import bilinear_zpk, zpk2tf, zpk2sos, freqs, sosfilt
import libtfr


Weighting = Enum("Weighting", ["A", "B", "C"])


def analog_zpk(weighting: Weighting) -> Tuple[float, float, float]:
    """Design an analog weighting filter with A, B, or C curve.

    Returns (zeros, poles, gain) of the filter.
    """
    # ANSI S1.4-1983 C weighting
    #    2 poles on the real axis at "20.6 Hz" HPF
    #    2 poles on the real axis at "12.2 kHz" LPF
    #    -3 dB down points at "10^1.5 (or 31.62) Hz"
    #                         "10^3.9 (or 7943) Hz"
    #
    # IEC 61672 specifies "10^1.5 Hz" and "10^3.9 Hz" points and formulas for
    # derivation.  See _derive_coefficients()
    pi = np.pi

    z = [0, 0]
    p = [
        -2 * pi * 20.598997057568145,
        -2 * pi * 20.598997057568145,
        -2 * pi * 12194.21714799801,
        -2 * pi * 12194.21714799801,
    ]
    k = 1

    if weighting is Weighting.A:
        # ANSI S1.4-1983 A weighting =
        #    Same as C weighting +
        #    2 poles on real axis at "107.7 and 737.9 Hz"
        #
        # IEC 61672 specifies cutoff of "10^2.45 Hz" and formulas for
        # derivation.  See _derive_coefficients()
        p.append(-2 * pi * 107.65264864304628)
        p.append(-2 * pi * 737.8622307362899)
        z.append(0)
        z.append(0)
    elif weighting is Weighting.B:
        # ANSI S1.4-1983 B weighting
        #    Same as C weighting +
        #    1 pole on real axis at "10^2.2 (or 158.5) Hz"
        p.append(-2 * pi * 10**2.2)  # exact
        z.append(0)
    elif weighting is Weighting.C:
        pass

    # Normalize to 0 dB at 1 kHz for all curves
    z = np.asarray(z)
    p = np.asarray(p)
    b, a = zpk2tf(z, p, k)
    k /= abs(freqs(b, a, [2 * pi * 1000])[1][0])

    return (z, p, k)


def A_weighting(fs: float, output="ba"):
    """Designs a digital A-weighting filter.

    Warning: fs should normally be higher than 20 kHz. For example,
    fs = 48000 yields a class 1-compliant filter.

    Parameters
    ----------
    fs : float
        Sampling frequency
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba'.

    """
    z, p, k = analog_zpk(Weighting.A)

    # Use the bilinear transformation to get the digital filter.
    z_d, p_d, k_d = bilinear_zpk(z, p, k, fs)

    if output == "zpk":
        return z_d, p_d, k_d
    elif output in {"ba", "tf"}:
        return zpk2tf(z_d, p_d, k_d)
    elif output == "sos":
        return zpk2sos(z_d, p_d, k_d)
    else:
        raise ValueError("'%s' is not a valid output form." % output)


class AWeightTransform:
    def __init__(self, sampling_rate: float):
        self.sos = A_weighting(sampling_rate, "sos")

    def transform(self, data):
        return sosfilt(self.sos, data)


class SpectrogramTransform:
    """Transforms a time series into a spectrogram using a hanning window."""

    def __init__(self, window_size: float, sampling_rate: float, max_frequency: float):
        self.step_size = window_size / 2
        self.sampling_rate = sampling_rate
        self.nfft = int(window_size * sampling_rate)
        self.freq_res = sampling_rate / self.nfft
        self.nstep = int(self.step_size * sampling_rate)
        window = np.hanning(self.nfft)
        self.scale1 = window.sum() ** 2
        self.scale2 = sampling_rate * (window**2).sum()
        self.enbw = sampling_rate * self.scale2 / self.scale1
        self.mfft = libtfr.mfft_precalc(self.nfft, window)
        self.freq, self.fidx = libtfr.fgrid(
            self.sampling_rate, self.nfft, (0, max_frequency)
        )

    def transform(self, data, scaling: str = "density"):
        spec = self.mfft.mtspec(data, self.nstep)[self.fidx]
        if scaling == "density":
            spec /= self.scale2
        elif scaling == "spectrum":
            spec /= self.scale1
        return spec

    def tgrid(self, spec):
        return libtfr.tgrid(spec, self.sampling_rate, self.nstep)
