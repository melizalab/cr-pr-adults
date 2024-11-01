# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Functions to perform filtering in the time domain

This code is mostly from https://github.com/endolith/waveform_analysis
"""
from enum import Enum
from typing import Tuple

import numpy as np
from scipy.signal import bilinear_zpk, zpk2tf, zpk2sos, freqs


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
