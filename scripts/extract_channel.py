#!/usr/bin/env python
# -*- mode: python -*-
"""Extract metadata from unit pprox files"""
import json
import logging
from pathlib import Path

from dlab import nbank


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug", help="show verbose log messages", action="store_true"
    )
    parser.add_argument(
        "responses", type=Path, help="directory with units to analyze (pprox format)"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        format="%(message)s", level=logging.DEBUG if args.debug else logging.INFO
    )
    print("unit,channel")
    for fname in args.responses.glob("*.pprox"):
        resource_name = fname.stem
        with open(fname) as jfp:
            pprox = json.load(jfp)
            channel = pprox.get("kilosort_source_channel")
            print(f"{resource_name},{channel}")


if __name__ == "__main__":
    main()
