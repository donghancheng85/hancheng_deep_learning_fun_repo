import torch
from torch import nn # nn contains all PyTorch building blocks for neural network (graphs)
import matplotlib.pyplot as plt # visualization

# PyTorch Work Flow
covered_in_this_section = {
    1: "data (prepare and load)",
    2: "build model",
    3: "fitting the model to data (training)",
    4: "make predications and evaluting a model (inference)",
    5: "saving and loading a model",
    6: "put everyting together"
}

print(covered_in_this_section)

print(f"torch version is {torch.__version__}")
