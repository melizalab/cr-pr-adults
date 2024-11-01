# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Divides a long recording into segments and calculates RMS amplitude and spectral
entropy in each segment.

The script calibrates its RMS measurements to dB SPL by attempting to find a
segment with a 1 kHz calibration tone of known amplitude.

The amplitude and spectral entropy measures are calculated by passing each
segment through a short-time Fourier transform and computing total power and
entropy in each frame. The statistics (mean, standard deviation, max) are
aggregated over the segment and output to a CSV file, one row per segment.

The temporal modulation spectra are calculated by taking the Fourier transform
of the amplitude envelopes in each segment. These are output to a raw binary
file as an double-precision array with dimensions n_mod_freqs by n_segments.

"""
import os
import argparse
import datetime
import logging
import csv
from pathlib import Path
from typing import NamedTuple, Optional

import arf
import h5py
from tqdm import tqdm
import numpy as np
from scipy import signal
import libtfr

from core import setup_log
from filters import A_weighting

# disable locking - neurobank archive is probably on an NFS share
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

log = logging.getLogger()  # root logger
__version__ = "20240508-1"

calibration_frequency_hz = 1000.0
# this is the amplitude measured at the opening of the calibrator. It's 94 dB
# SPL inside the opening but quite a bit less where the recording mic is
# positioned when I record the calibration signal.
calibration_amplitude_dB = 80.0

amplitude_quantiles = [0, 0.25, 0.5, 0.75, 1.0]
amplitude_quantile_names = [f"ampl_q{q*100:.0f}" for q in amplitude_quantiles]


class Segment(NamedTuple):
    time: datetime.datetime
    dataset: h5py.Dataset
    start_sample: int
    end_sample: int


class SegmentIterator:
    """Divides a collection of ARF files into segments. For performance
    reasons, all the segments have to have the same sampling rate.

    """

    sampling_rate = None
    calibration_segment = None
    segments = []

    def __init__(self, files, segment_size: float):
        segment_shift = segment_size / 2
        for file in files:
            handle = h5py.File(file, "r")
            # the calibration recording should be the first one; if there are
            # multiple files they may have their own copies of the calibration,
            # but this will always be first created.
            for entry_name in arf.keys_by_creation(handle):
                entry = handle[entry_name]
                if not arf.is_entry(entry) or "pcm_000" not in entry:
                    log.debug("- %s/%s: not a recording, skipping", file, entry_name)
                    continue
                dset = entry["pcm_000"]
                entry_start = arf.timestamp_to_datetime(entry.attrs["timestamp"])
                # if no explicitly labeled calibration recording, use the first entry
                if self.calibration_segment is None:
                    log.info("- %s/%s: using for calibration", file, entry_name)
                    self.calibration_segment = Segment(
                        time=entry_start,
                        dataset=dset,
                        start_sample=0,
                        end_sample=dset.size,
                    )
                    self.sampling_rate = dset.attrs["sampling_rate"]
                elif entry_name == "calibration":
                    log.debug(
                        "- %s/%s: already found a calibration recording, skipping",
                        file,
                        entry_name,
                    )
                else:
                    dset_offset = dset.attrs.get("offset", 0)
                    if dset.attrs["sampling_rate"] != self.sampling_rate:
                        raise ValueError(
                            "- all recordings must have the same sampling rate"
                        )
                    nsamples = int(segment_size * self.sampling_rate)
                    nshift = int(segment_shift * self.sampling_rate)
                    i = 0
                    for i, segment_end in enumerate(range(nsamples, dset.size, nshift)):
                        segment_start = segment_end - nsamples
                        self.segments.append(
                            Segment(
                                time=entry_start
                                + datetime.timedelta(
                                    seconds=(dset_offset + segment_start)
                                    / self.sampling_rate
                                ),
                                dataset=dset,
                                start_sample=segment_start,
                                end_sample=segment_end,
                            )
                        )
                    log.debug("- %s/%s: %d segments", file, entry_name, i)
        log.info("- found a total of %d segments", len(self.segments))

    def __len__(self):
        return len(self.segments)

    def __iter__(self):
        return iter(self.segments)


class AWeightTransform:
    def __init__(self, sampling_rate: float):
        self.sos = A_weighting(sampling_rate, "sos")

    def transform(self, data):
        return signal.sosfilt(self.sos, data)


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


class MPSTransform:
    def __init__(self, segment_size: float, spec_shift: float, max_frequency: float):
        self.spec_shift = spec_shift
        self.segment_nframes = int(segment_size / spec_shift)
        # preallocate the fft for MPS calculation, because all segments will have
        # the same number of frames regardless of sampling rate.
        mps_window = np.hanning(self.segment_nframes)
        self.scale = mps_window.sum() ** 2
        log.info(
            "Allocating %d-point FFT transform for modulation power spectrum",
            self.segment_nframes,
        )
        self.mfft = libtfr.mfft_precalc(self.segment_nframes, mps_window)
        self.freq, self.fidx = libtfr.fgrid(
            1 / self.spec_shift, self.segment_nframes, (0, max_frequency)
        )
        self.tgrid = np.arange(
            0, self.segment_nframes * self.spec_shift, self.spec_shift
        )

    def transform(self, data):
        spec = self.mfft.mtpsd(data) / self.scale
        return spec[self.fidx]

    @property
    def size(self):
        return self.fidx.size


def get_calibration(segment: Segment, window_size: float) -> float:
    log.info(
        "Determining calibration from %s%s using flattop window",
        segment.dataset.file.filename,
        segment.dataset.name,
    )
    sampling_rate = segment.dataset.attrs["sampling_rate"]
    nfft = int(window_size * sampling_rate)
    df = sampling_rate / nfft
    step_size = window_size * 0.782
    nstep = int(step_size * sampling_rate)
    # flattop windows are better for estimating amplitude of sinusoids
    window = signal.windows.flattop(nfft)
    scale = window.sum() ** 2  # gives us power in units of V**2
    mfft = libtfr.mfft_precalc(nfft, window)
    freqs, _ = libtfr.fgrid(sampling_rate, nfft)

    data = segment.dataset[segment.start_sample : segment.end_sample]
    spec = mfft.mtspec(data, nstep) / scale
    ps = spec.mean(1)
    peak_idx = ps.argmax()
    peak_freq = freqs[peak_idx]
    if peak_freq < (calibration_frequency_hz * 0.9) or peak_freq > (
        calibration_frequency_hz * 1.1
    ):
        raise ValueError(
            "Unable to detect calibration tone in {segment.dataset.file.filename}{segment.dataset.name}"
        )
    recorded_dB = 10 * np.log10(ps[peak_idx])
    # here we use 20 because the correction is on the amplitude scale
    dBSPL_correction = 10 ** ((calibration_amplitude_dB - recorded_dB) / 20)
    log.info(
        "- found calibration tone (%2.f dB SPL) with peak of %.2f dB FS at %.2f Hz.",
        calibration_amplitude_dB,
        recorded_dB,
        peak_freq,
    )
    log.info("- recordings will be rescaled by %.0fx to get dB SPL.", dBSPL_correction)
    return dBSPL_correction


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Calculate acoustical statistics of recording(s)"
    )
    p.add_argument(
        "-v", "--version", action="version", version="%(prog)s " + __version__
    )
    p.add_argument("--debug", help="show verbose log messages", action="store_true")
    p.add_argument(
        "--segment-size",
        type=float,
        default=600.0,
        help="size of segments to analyze (default %(default)s s)",
    )
    p.add_argument(
        "--window-size",
        type=float,
        default=0.010,
        help="spectrogram window size (default %(default)s s)",
    )
    p.add_argument(
        "--max-frequency",
        type=float,
        default=8000.0,
        help="maximum spectral frequency to consider (default %(default).0f Hz)",
    )
    p.add_argument("input", type=Path, nargs="+", help="the input ARF file(s)")
    p.add_argument(
        "output", type=Path, help="the output ARF file (overwrites existing files)"
    )
    args = p.parse_args()
    setup_log(log, args.debug)

    log.info("Parsing input files:")
    segments = SegmentIterator(args.input, args.segment_size)

    SPL_calibration = get_calibration(segments.calibration_segment, args.window_size)

    # do some preallocations
    make_spectrogram = SpectrogramTransform(
        args.window_size, segments.sampling_rate, args.max_frequency
    )
    a_weighter = AWeightTransform(segments.sampling_rate)

    log.info("- acoustical statistics -> %s", args.output)
    fieldnames = ["entry", "date", "time", "ampl_avg"] + amplitude_quantile_names
    with open(args.output, "w") as csv_fp:
        writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
        writer.writeheader()
        for i, segment in enumerate(tqdm(segments)):
            # NB segments can vary in length
            data = (
                segment.dataset[segment.start_sample : segment.end_sample]
                * SPL_calibration
            )
            filtered_data = a_weighter.transform(data)
            spec = make_spectrogram.transform(filtered_data, scaling="density")
            nfreq, nframes = spec.shape
            power_envelope = spec.sum(0) * make_spectrogram.freq_res
            amplitude_envelope = 10 * np.log10(power_envelope)
            ampl_mean = 10 * np.log10(power_envelope.mean())
            ampl_quantiles = dict(
                zip(
                    amplitude_quantile_names,
                    10 * np.log10(np.quantile(power_envelope, amplitude_quantiles)),
                )
            )
            writer.writerow(
                {
                    "entry": segment.dataset.parent.name,
                    "date": segment.time.date(),
                    "time": segment.time.time(),
                    "ampl_avg": ampl_mean,
                }
                | ampl_quantiles
            )
