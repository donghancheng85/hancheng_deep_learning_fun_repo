# Brief

Hancheng's repo for having fun in the Deep Learning world. Will update more later!!


## Daily setup on Ubuntu + RTX 5080 machine
```bash
source .venv/bin/activate
```

## Development Setup (Ubuntu 24.04 + venv + NVIDIA GPU)

This repo uses a per-project Python virtual environment (`.venv`) and pip.

### Prereqs
- Ubuntu 24.04
- NVIDIA driver installed and working (`nvidia-smi` shows GPU)
- Python 3.12 (default on Ubuntu 24.04)

Install system packages:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git
```

### Create and activate venv

```bash
cd /path/to/hancheng_deep_learning_fun_repo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

### Install PyTorch (RTX 5080 / sm_120)

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Install remaining dependencies

```bash
pip install -r requirements.txt
```

### Verify GPU works

```bash
python common/verify_gpu.py
```

Expected sample output:
```bash
torch: 2.11.0.dev20260131+cu128
cuda available: True
gpu: NVIDIA GeForce RTX 5080
arch list: ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
matmul ok, mean: 0.0392305813729763
```

## Development Setup (conda + CPU/MPS)

Activate the ENV for this project.
```bash
conda env create -f environment.conda.yml
conda activate hancheng-dl
```


## Run code (venv + RTX5080)

Since this repo is structured as a learning repo. To use the package in common (by `from common.*** import ***`), please add `PYTHONPATH=.` before directly runnig a code using `python`

### Create an alias in venv for `PYTHONPATH=. python`

A the end of `.venv/bin/activate` add the following lines:

```bash
# ---- project helper: pyp ---------------------------------
hancheng_dl_python() {
    PYTHONPATH="$(pwd)" python "$@"
}
# ----------------------------------------------------------

```

In this case, the defined alias is `hancheng_dl_python`.

Running `hancheng_dl_python` = `PYTHONPATH=. python`


## Run code (conda + CPU/MPS)

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
