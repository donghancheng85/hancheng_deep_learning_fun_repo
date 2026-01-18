# Brief

Hancheng's repo for having fun in the Deep Learning world. Will update more later!!

## Setup

Activate the ENV for this project.
```bash
conda env create -f environment.yml
conda activate hancheng-dl
```

## Run code

Since this repo is structured as a learning repo. To use the package in common (by `from common.*** import ***`), please add `PYTHONPATH=.` before directly runnig a code using `python`

e.g., running [g_25_get_tensor_attributes](lessons/section2_pytorch_fundamentals/src/g_25_get_tensor_attributes.py)

```bash
PYTHONPATH=. python lessons/section2_pytorch_fundamentals/src/g_25_get_tensor_attributes.py
```

### Create an alias in Conda ENV for the `PYTHONPATH=. python`

Suppose the ENV name is `hancheng-dl`

1. Run the following command
```bash
conda activate hancheng-dl # need to in the ENV first to access the environment variable

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
```

2. Add alias on environment activation
Run and created
```bash
nano $CONDA_PREFIX/etc/conda/activate.d/aliases.sh
```

Then add the following, suppose the created alias name is `hancheng_dl_python`
```bash
# Alias runs inside hancheng-dl
alias hancheng_dl_python='PYTHONPATH=. python'
```

3. Remove alias on environment deactivation

This step is necessary since we do not want this alias to affect other ENVs

Run and create:
```bash
nano $CONDA_PREFIX/etc/conda/deactivate.d/aliases.sh
```

Add the following content so the alias `hancheng_dl_python` will be deactivated after we deactivated `hancheng-dl`
```bash
unalias hancheng_dl_python 2>/dev/null || true
```
