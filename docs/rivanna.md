
## Rivanna:

A number of analyses can take a long time to run and benefit greatly from using a high-performance computing cluster. These instructions are for the UVA HPC cluster but should be possible to adapt to other systems using SLURM. Some of the top-level module dependencies may have changed, but the code should still run if they are updated.

Note that you will need to rsync the `build` directory to the HPC cluster, as the batch jobs expect the spike response or behavioral data to be there.

### Setup

Start an interactive session:

`ijob -c 2 -A melizalab -p standard --time 1:00:00`

Activate the required modules and install the dependencies using anaconda and pip.

``` shell
module load anaconda/2023.07-py3.11
conda create --name cr-pr-adults python=3.11 scikit-learn-intelex pandas
conda activate cr-pr-adults
python -m pip install -e .
```

Test that the script works:

``` shell
python scripts/motif_discrim.py -o temp C42_3_1_c156
```

### Batch: Motif discrimination

Run `sbatch batch/motif_discrim.slurm` to run `scripts/motif_discrim.py` on all units.

The motif discrimination batch allocates an hour to each job, which may not be enough time for some units. To check for which units failed to complete analysis in time, you can use this bash command:

``` shell
grep -l CANCELLED logfiles/*.log | grep -Eo '[0-9]+' | sort -g | tr '\n' ','
```

Copy the list of failed jobs and rerun them as follows with a 2-hour limit:

``` shell
sbatch --array <list-of-jobs> -t 2:00:00 batch/motif_discrim.slurm
```

### Batch: Behavioral state space model

Run `sbatch batch/ssm-pretraining.slurm` to analyze each subject's behavioral data.
