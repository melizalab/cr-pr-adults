# CR-PR-adults

This project examines how the social-acoustical environment a zebra finch
experiences during childhood influences auditory processing and perception in
adulthood.

This file describes the key steps in running the analysis, including links to
data files hosted on public repositories.. See the `docs` folder for information
about installing dependencies, using Rivanna, and other topics.

## Metadata

- `inputs/bird_metadata.csv`: for all the birds included in the study, this table has the name, unique identifier, sex, rearing condition, age (at start of recording or training), and number of siblings.
- `inputs/recording_metadata.csv`: for each electrophysiology recording, this table has the hemisphere and brain region
- `inputs/all_units.tbl`: a list of single units included in the analysis, one per line. Each name refers to two neurobank resources, one with the spike response data (in pprox format) and one with the spike waveforms (in hdf5 format).
- `inputs/behavior_data.tbl`: a table of behavioral datasets to retrieve from decide-host

## Datasets

- amplitude and texture statistics for colony and pair-reared birds
- behavior trial data
- neural data

## Analysis

Temporary files are created in the `build` directory. You may need to create this first if it doesn't already exist. To restart the analysis from the beginning, you can just clear this directory out.

The code is a combination of scripts and Jupyter notebooks. You typically need to run the scripts first and then the notebooks.

If you're starting from a fresh repository, see `docs/installation.md` for instructions about how to set up your Python and R environments. There are some steps that more or less need to be run on a high-performance computing cluster. See `docs/rivanna.md` for the steps we use to set this up on UVA's HPC cluster Rivanna.

### Acoustical environment statistics

The amplitude statistics for the colony and pair settings are in the `zebf-social-acoustical-stats` dataset as CSV files. The texture statistics (McDermott and Simoncelli 2011) need to be calculated from the sample wave files. There are instructions in the `notebooks/figure1_acoustical-stats.ipynb` Jupyter notebook.

## Behavior

The behavior trial data are in the `zebf-discrim-noise` dataset as CSV files. They were retrieved from our internal trial data on 2024-10-12 using the command `batch/retrieve_trials.sh < inputs/behavior_data.tbl`. The `retrieve_trials` script is retained in this code repository for reference but will not work from outside our internal network.

The training data for an example bird is analyzed using a state space model. This model is fit using a separate script, `scripts/ssm-training.R`. Instructions for how to fit the model and code to generate the panels in Figure 2 are in the `notebooks/figure2_behavior-stats.ipynb` notebook.

Analysis and figure generation are run in Jupyter notebooks. See `docs/installation.md` for instructions about setting this up.

- `notebooks/behavior-pretraining.ipynb`: R notebook with basic visualizations for behavior
- `notebooks/behavior-ssm-summary.ipynb`: R notebook with visualizations for behavior state-space models
- `notebooks/spike-waveforms.ipynb`: Python notebook to plot spike waveform classification panels

## Electrophysiology

### Initial preprocessing

1. `scripts/unit_waveforms.py -o build inputs/all_units.tbl` to classify units as narrow or wide-spiking. Outputs mean spike waveforms to `build/mean_spike_waveforms.csv` and classifications to `build/mean_spike_features.csv`.
2. `batch/motif_rates.sh < inputs/all_units.tbl` to compute average firing rates for each motif in the stimulus set. Outputs to `build/*_rates.csv` files.
3. `batch/motif_discrim.sh < inputs/all_units.tbl` to compute auditory responsiveness. This can (should) be run on rivanna using `batch/motif_discrim.slurm`. See `docs/rivanna.md` for notes about setting this up.
4. `batch/pairwise_correlations.sh < inputs/recording_metadata.csv`

### Single-Unit Analysis

1. `notebooks/spike-waveforms-figure.ipynb` generates Figure 2B. 
2. `notebooks/motif-resp-example-figure.ipynb` generates Figure 2C (example raster plots) and Figure 5A-C (example discriminability and selectivity plots).
3. `notebooks/single-unit-analysis.ipynb` will run a GLM for each unit to determine how many motifs evoke significant responses and generates the summary panels in Figures 3 and 5. It also generates the `build/cr_units.txt` and `build/pr_units.txt` control files used by the decoder.
4. `notebooks/pairwise-corr.ipynb` generates panels for Figure 4.

### Decoder Analysis

Individual decoder models can be fit using `scripts/decoder.py`. For example, `venv/bin/python scripts/decoder.py -o build build/cr_units.txt` will load the data for all the CR units, compute average firing rates and delay-embed them, use cross-validation to determine the optimal ridge regression penalty hyperparameter, and then do train/test splits for each motif to compute predictions from the responses to clean and noise-embedded stimuli.

The full analysis (Figure 6C) requires doing this multiple times for subsamples CR and PR populations, so it should be run on an HPC cluster (see `docs/rivanna.md` for how we do this at UVA)

1. The first step is to generate a table of jobs to run. `venv/bin/python scripts/make_decoder_tasks.py > inputs/decoder_tasks.txt` will read in `build/cr_units.txt` and `build/pr_units.txt` and make this table, with 100 replicates for 15 different ensemble sizes.
2. On your HPC cluster, run `sbatch batch/decoder.slurm`
3. Collate the CSV outputs into a single file: `awk 'FNR==1 && NR!=1{next;}{print}' build/*_model.csv  > build/decoder_predictions.csv`
4. `notebooks/decoder-summary.ipynb` generates 


### Optional/Deprecated

4. (optional) `batch/plot_rasters.sh < inputs/all_units.tbl` to generate inspection plots for all units
3. `batch/extract_recording_channel.sh > build/unit_channels.csv` to extract the primary electrode channel for each unit
5. `batch/motif_distances.sh < inputs/all_units.tbl` to measure how well responses of single units to different motifs can be discriminated using a spike-distance metric
6. `batch/sequence_distances.sh < inputs/all_units.tbl` to measure how consistently neurons respond to sequences of motifs in the presence of noise
