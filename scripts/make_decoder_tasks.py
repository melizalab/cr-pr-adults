#!/usr/bin/env python
# -*- mode: python -*-
""" Generate a table of tasks for the decoder batch """
from pathlib import Path

import numpy as np


def count_units(path):
    return len(path.read_text().splitlines())


input_dir = Path("build")
unit_lists = ["cr_units.txt", "pr_units.txt"]
min_units = 40
n_conditions = 15
n_replicates = 100
initial_seed = 1024

max_dataset_units = [count_units(input_dir / p) for p in unit_lists]
max_units = min(max_dataset_units)
n_units = (
    np.logspace(np.log2(min_units), np.log2(max_units), num=n_conditions, base=2)
    .round()
    .astype("i")
)

for unit_list, max_units in zip(unit_lists, max_dataset_units):
    for n_subsample in n_units:
        if n_subsample == max_units:
            continue
        for seed in range(initial_seed, initial_seed + n_replicates):
            print(f"{input_dir / unit_list}  {n_subsample}  {seed}")

for unit_list, max_units in zip(unit_lists, max_dataset_units):
    print(f"{input_dir / unit_list}  {max_units}  {initial_seed}")
