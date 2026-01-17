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
