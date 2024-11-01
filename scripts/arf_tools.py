# -*- coding: utf-8 -*-
# -*- mode: python -*-
import datetime
import logging
from typing import NamedTuple, Optional

import h5py
import arf

from core import Waveform

log = logging.getLogger("colony-noise")  # root logger


class Segment(NamedTuple):
    time: datetime.datetime
    dataset: h5py.Dataset
    start_sample: int
    end_sample: int


def find_segment(
    arf_file, time: datetime.datetime, segment_size: float
) -> Optional[Segment]:
    for entry_name, entry in arf_file.items():
        if not arf.is_entry(entry) or "pcm_000" not in entry:
            log.debug("- %s: not a recording, skipping", entry_name)
            continue
        entry_start = arf.timestamp_to_datetime(entry.attrs["timestamp"])
        dset = entry["pcm_000"]
        sampling_rate = dset.attrs["sampling_rate"]
        entry_end = entry_start + datetime.timedelta(seconds=dset.size / sampling_rate)
        log.debug("- %s: spans %s--%s", entry_name, entry_start, entry_end)
        if time < entry_end:
            segment_start = (time - entry_start).total_seconds() * sampling_rate
            segment_end = min(segment_start + segment_size * sampling_rate, dset.size)
            return Segment(time, dset, int(segment_start), int(segment_end))


def dataset_to_segment(dataset):
    entry = dataset.parent
    entry_start = arf.timestamp_to_datetime(entry.attrs["timestamp"])
    return Segment(entry_start, dataset, 0, dataset.size)


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
