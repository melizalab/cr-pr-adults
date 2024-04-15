# -*- coding: utf-8 -*-
# -*- mode: python -*-

import matplotlib as mpl

tickparams = {
    "major.size": 2,
    "major.pad": 1.5,
    "minor.size": 1,
    "labelsize": "small",
    "direction": "out",
}
grparams = {
    "font": {"size": 6},
    "axes": {"linewidth": 0.5, "unicode_minus": False, "titlesize": 6},
    "lines": {"linewidth": 0.5},
    "xtick": tickparams,
    "ytick": tickparams,
    "image": {"aspect": "auto", "origin": "lower"},
    "pdf": {"fonttype": 42},
}

RANDOMSEED = 10024

for k, v in grparams.items():
    mpl.rc(k, **v)
