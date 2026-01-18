import pandas as pd
import numpy as np
import torch
import sklearn
import matplotlib
import torchinfo, torchmetrics


def test():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    x = torch.randn(3, 3, device=device)
    print("torch:", torch.__version__)
    print("device:", device)
    print(x @ x)

    print(f"is cuda avaliable? -- {torch.cuda.is_available()}")
    print(f"Is mps avaliable? {torch.backends.mps.is_available()}")
