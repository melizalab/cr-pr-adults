# CR-PR-adults

This project examines how the social-acoustical environment a zebra finch
experiences during childhood influences auditory processing and perception in
adulthood.

This file describes the key steps in running the analysis, including links to
data files hosted on public repositories. See the `docs` folder for information
about installing dependencies, using a high performance cluster, and other
topics. The instructions should work on Linux or any other POSIX-compatible
operating system. Windows users will need to port batch scripts.

## Datasets

The data for the analysis has been deposited as zip files in three figshare datasets

- zebf-social-acoustical-stats: 10.6084/m9.figshare.27961518
- zebf-discrim-noise: 10.6084/m9.figshare.27961002
- zebf-social-acoustical-ephys: 10.6084/m9.figshare.27961362

To download, verify, and unpack the datasets, run `batch/fetch_datasets.sh`. 

## Analysis

Temporary files are created in the `build` directory. You may need to create this first if it doesn't already exist. To restart the analysis from the beginning, you can just clear this directory out.

The code is a combination of scripts and Jupyter notebooks. You typically need to run the scripts first and then the notebooks.

If you're starting from a fresh repository, see `docs/installation.md` for instructions about how to set up your Python and R environments. There are some steps that more or less need to be run on a high-performance computing cluster. See `docs/rivanna.md` for the steps we use to set this up on UVA's HPC cluster Rivanna.

### Acoustical environment statistics

The amplitude statistics for the colony and pair settings are in the `zebf-social-acoustical-stats` dataset as CSV files. The texture statistics (McDermott and Simoncelli 2011) need to be calculated from the sample wave files. There are instructions in the `notebooks/figure1_acoustical-stats.ipynb` Jupyter notebook.

## Behavior

The behavior trial data are in the `zebf-discrim-noise` dataset as CSV files. They were retrieved from our internal trial data on 2024-10-12 using the command `batch/retrieve_trials.sh < inputs/behavior_data.tbl`. The `retrieve_trials` script is retained in this code repository for reference but will not work from outside our internal network.

The training data for an example bird is analyzed using a state space model. This model is fit using a separate script, `scripts/ssm-training.R`. Instructions for how to fit the model and code to generate the panels in Figure 2 are in the `notebooks/figure2_behavior-stats.ipynb` notebook.

Spectrograms for example behavioral stimuli can be plotted with `notebooks/figure2_behavior-stims.ipynb`.

## Electrophysiology

The `zebf-social-acoustical-ephys` dataset contains the extracellular spike times recorded from the auditory pallium; metadata about the birds, recording sites, and spike waveforms; metadata about the stimuli (used to split long sequences up into individual motifs); and sound files with the stimuli presented during the recording.

### Initial preprocessing

Spike waveforms were saved during spike sorting by `group-kilo-spikes`, a script in our collection of custom Python code (https://github.com/melizalab/melizalab-tools). We used the command `scripts/unit_waveforms.py -o build inputs/all_units.tbl` to upsample, align, and average waveforms for each unit and to classify units as narrow or wide-spiking. The outputs are stored in the dataset, but the script is retained for reference.

First, extract spike rates for each of the 10 stimulus motifs at different SNR levels using the `scripts/motif_rates.py` script. This can be run as a batch using `batch/motif_rates.sh < inputs/all_units.tbl`, which will output a CSV file for each unit to the `build` folder.

Second, calculate selectivity from the rates generated in the previous step using a generalized linear model. We do this in R because we need to get the estimated marginal means. Run `Rscript scripts/motif_selectivity.R`. This step outputs `build/cr_units.txt` and `build/pr_units.txt`, which are needed for the decoder analysis.

Third, calculate discriminability using `scripts/motif_discrim.py`. This can be run as a batch using `batch/motif_discrim.sh < inputs/all_units.tbl`, but this can take a very long time because we run a permutation test for each unit to see if the discriminability is greater than expected from chance. If you have access to an HPC cluster, you may be able to adapt `batch/motif_discrim.slurm` for your job control system.

Fourth, calculate pairwise correlations using `scripts/pairwise_correlations.py`. It reads in rates from the files in `build` generated by `scripts/motif_rates.py` and outputs the correlations to this directory. The batch command for this is `batch/pairwise_correlations.sh < datasets/zebf-social-acoustical-ephys/metadata/recordings.csv`.

Finally, extract the source channels from the pprox files by running `scripts/extract_channel.py datasets/zebf-social-acoustical-ephys/responses/ > build/unit_channels.csv`

### Single-Unit Analysis

1. `notebooks/figure3_ephys-examples.ipynb` generates plots of waveform shape and example responses to songs
2. `notebooks/figure4-6_single-unit-stats.ipynb` generates summary plots for firing rates (Fig 4) and discriminability and selectivity (Fig 6). Make sure to run this before subsequent notebooks.
3. `notebooks/figure5_pairwise-corr.ipynb` generates summary plots for the pairwise correlation analysis
4. `notebooks/figure6_discriminability.ipynb` generates example plots for the discriminability analysis
5. `notebooks/figure6_selectivity.ipynb` generates example plots for the selectivity analysis

### Decoder Analysis

Individual decoder models can be fit using `scripts/decoder.py`. For example, `venv/bin/python scripts/decoder.py -o build build/cr_units.txt` will load the data for all the CR units, compute average firing rates and delay-embed them, use cross-validation to determine the optimal ridge regression penalty hyperparameter, and then do train/test splits for each motif to compute predictions from the responses to clean and noise-embedded stimuli.

The full analysis (Figures 7 and 8) requires doing this multiple times for subsamples of the CR and PR populations, so it should be run on an HPC cluster (see `docs/rivanna.md` for how we do this at UVA)

1. The first step is to generate a table of jobs to run. `venv/bin/python scripts/make_decoder_tasks.py > inputs/decoder_tasks.txt` will read in `build/cr_units.txt` and `build/pr_units.txt` and make this table, with 100 replicates for 15 different ensemble sizes.
2. On your HPC cluster, run `sbatch batch/decoder.slurm`
3. Collate the CSV outputs into a single file: `awk 'FNR==1 && NR!=1{next;}{print}' build/*_model.csv  > build/decoder_predictions.csv`
4. `notebooks/figure7-8_decoder-example.ipynb` generates example panels for the decoder analysis
4. `notebooks/figure7-8_decoder-summary.ipynb` generates panels of the summary statistics for the decoder
