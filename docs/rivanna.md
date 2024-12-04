
## Rivanna:

A number of analyses can take a long time to run and benefit greatly from using a high-performance computing cluster. These instructions are for the UVA HPC cluster but should be possible to adapt to other systems that use SLURM. Some of the top-level module dependencies may have changed, but the code should still run if they are updated.

Note that if you do preprocessing on another machine, you need to copy the `build` directory to the HPC cluster, and then copy it back to your main analysis machine.

### Setup

Start an interactive session:

``` shell
ijob -c 2 -A melizalab -p standard --time 1:00:00
```

Activate the required modules and install the dependencies using anaconda and pip:

``` shell
module load miniforge/24.3.0-py3.11 parallel/20200322
mamba create --name cr-pr-adults python=3.11 scikit-learn-intelex pandas
source activate cr-pr-adults
python -m pip install -r requirements.txt
```

To test that everything works before firing up a big batch, run a job in the interactive session. For example:

``` shell
python scripts/motif_rates.py -o build -m datasets/zebf-social-acoustical-ephys/metadata -o build datasets/zebf-social-acoustical-ephys/responses/C294_1_1_c14.pprox
```

### Batch: general

You should be able to run any of the batch scripts within your interactive session as long
as you point them to the right python interpreter. Start a new session with a bunch of cores, then run the following commands before invoking the batch scripts.

``` shell
module load miniforge/24.3.0-py3.11 parallel/20200322
source activate cr-pr-adults
export PYTHON=$(which python)
```

If you need to start a new session, just load the miniforge module and activate your conda environment.

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

### R scripts

For R scripts, you'll use an interactive session, but you need to load the R module rather than Python. See `installation.md` for instructions about installing R dependencies.

``` shell
module load gcc/11.4.0 openmpi/4.1.4 R/4.3.1
```
