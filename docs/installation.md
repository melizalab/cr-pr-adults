
## Installation

If you've just cloned the repository, you'll need to set up a virtual environment to bring in some dependencies. These steps only need to be run once.

``` shell
python3 -m venv --upgrade-deps venv
venv/bin/python -m pip install --upgrade pip setuptools wheel
```

If your goal is to reproduce the last good analysis, you should install dependencies from the `requirements.txt` file.

``` shell
venv/bin/python -m pip install -r requirements.txt
```

If you're doing development, you should instead install the most recent versions of all the dependencies:

``` shell
venv/bin/python -m pip install -e .
```

If you want to use Jupyter Lab, you'll need to register your virtual environment with the server. Assuming you already have jupyterlab installed (as a system package or using pipx):

``` shell
python -m pip install ipykernel  # this has to be done for any fresh virtual environment
python -m ipykernel install --user --name=cr-pr-adults  # only do this if you haven't registered the kernel before
```

Some of the analysis/plotting notebooks use R instead of Python. Run the following commands in R to register a kernel with Jupyter and to install dependencies:

``` R
install.packages(c('tidyverse', 'lme4', 'emmeans', 'ggplot2'))   # direct code dependencies
install.packages(c('repr', 'IRdisplay', 'IRkernel'))             # for the R notebooks
IRkernel::installspec(name = 'ir43', displayname = 'R 4.3')

```

## Running the code

In Jupyter Lab, you'll need to set the kernel for your notebook to `cr-pr-adults`. If you're running a script, make sure to activate your venv first (`source venv/bin/activate`).

See the [lab wiki](https://gracula.psyc.virginia.edu/wiki/Computing/RemotePythonGuide/) for instructions on how to run a persistent jupyter lab server for your project.
