# CR-PR-adults

This project examines how the social-acoustical environment a zebra finch
experiences during childhood influences auditory processing and perception in
adulthood. This is a spin-off of the noise-invariant-analysis project, but
starting from a clean git repository and only including scripts and notebooks
needed for the planned paper.

This file describes the key steps in running the analysis. See the `docs` folder
for information about installing dependencies, using Rivanna, and other topics.

## Key control files

- `inputs/bird_metadata.csv`: for all the birds included in the study, this table has the name, unique identifier, sex, rearing condition, age (at start of recording or training), and number of siblings.
- `inputs/recording_metadata.csv`: for each electrophysiology recording, this table has the hemisphere and brain region
- `inputs/all_units.tbl`: a list of single units included in the analysis, one per line. Each name refers to two neurobank resources, one with the spike response data (in pprox format) and one with the spike waveforms (in hdf5 format).
- `inputs/behavior_data.tbl`: a table of behavioral datasets to retrieve from decide-host

## Behavior

Preprocessing:

1. Run `batch/retrieve_trials.sh < inputs/behavior_data.tbl` to retrieve trials for all of the included subjects. Saves the output to `build/*_trials.csv` files, one for each line in `inputs/behavior_data.tbl`
2. Run `batch/ssm-pretraining.sh` to fit state-space models to the behavioral data. This can also be run on rivanna using `batch/ssm-pretraining.slurm`.

Analysis and figure generation are run in Jupyter notebooks. See `docs/installation.md` for instructions about setting this up.

- `notebooks/behavior-pretraining.ipynb`: R notebook with basic visualizations for behavior
- `notebooks/behavior-ssm-summary.ipynb`: R notebook with visualizations for behavior state-space models
- `notebooks/spike-waveforms.ipynb`: Python notebook to plot spike waveform classification panels

## Electrophysiology

1. `scripts/unit_waveforms.py -o build inputs/all_units.tbl` to classify units as narrow or wide-spiking. Outputs mean spike waveforms to `build/mean_spike_waveforms.csv` and classifications to `build/mean_spike_features.csv`.
2. `batch/motif_rates.sh < inputs/all_units.tbl` to compute average firing rates for each motif in the stimulus set. Outputs to `build/*_rates.csv` files.
3. `batch/motif_discrim.sh < inputs/all_units.tbl` to compute auditory responsiveness. This can (should) be run on rivanna using `batch/motif_discrim.slurm`. See `docs/rivanna.md` for notes about setting this up.
4. `batch/pairwise_correlations.sh < inputs/recording_metadata.csv`

### Optional/Deprecated

4. (optional) `batch/plot_rasters.sh < inputs/all_units.tbl` to generate inspection plots for all units
3. `batch/extract_recording_channel.sh > build/unit_channels.csv` to extract the primary electrode channel for each unit
5. `batch/motif_distances.sh < inputs/all_units.tbl` to measure how well responses of single units to different motifs can be discriminated using a spike-distance metric
6. `batch/sequence_distances.sh < inputs/all_units.tbl` to measure how consistently neurons respond to sequences of motifs in the presence of noise
