import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

"""
1. What are 3 areas in industry where computer vision is currently being used?
   - Autonomous vehicles (self-driving cars)
   - Healthcare (medical imaging and diagnostics)
   - Retail (inventory management and customer behavior analysis)
"""

"""
2. Search "what is overfitting in machine learning" and write down a sentence about what you find.
   - Overfitting in machine learning occurs when a model learns the training data too well, 
     capturing noise and details that do not generalize to new data.
"""

"""
3. Search "ways to prevent overfitting in machine learning", write down 3 of the things you find and a sentence about each. Note: there are lots of these, so don't worry too much about all of them, just pick 3 and start with those.
   - More data: Increasing the size of the training dataset can help the model learn more generalizable patterns and reduce overfitting.
   - Regularization: Adding a penalty to the loss function to constrain the model's complexity.
   - Dropout: Randomly setting a fraction of the input units to zero during training to prevent co-adaptation.
   - Early stopping: Monitoring the model's performance on a validation set and stopping training when performance starts to degrade.
"""

"""
4. Load the torchvision.datasets.MNIST() train and test datasets.
"""
train_data = datasets.MNIST(
    root="lessons/section5_pytorch_computer_vision/data",
    train=True,
    download=False,
    transform=ToTensor(),
    target_transform=None,  # labels are plain integers: e.g. 7
)

test_data = datasets.MNIST(
    root="lessons/section5_pytorch_computer_vision/data",
    train=False,
    download=False,
    transform=ToTensor(),
    target_transform=None,
)

class_name = train_data.classes
print(f"class names are: {class_name}")

"""
4.1 target_transform demonstration — comparison with and without

WITHOUT target_transform (target_transform=None):
  label = 5                              (plain Python int)
  type  = <class 'int'>

WITH target_transform (one-hot encode):
  label = tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.])  (float32 tensor)
  type  = <class 'torch.Tensor'>
  shape = torch.Size([10])

When is target_transform useful?
  - Custom loss functions that expect probability vectors instead of class indices
    (e.g. KL-Divergence loss, label-smoothing implementations done manually)
  - Multi-label classification where you want to map a single integer to a
    bitmask of active classes
  - Ordinal regression where you encode class k as k ones followed by zeros
  - Any scenario where the raw integer label needs to be converted BEFORE
    being passed to the model or loss function

NOTE: nn.CrossEntropyLoss (used in this project) expects plain integer labels,
so target_transform=None is the correct choice for standard classification.
"""

# --- Define a one-hot encoding transform ---
NUM_CLASSES = 10  # MNIST digits 0-9


def one_hot_encode(label: int) -> torch.Tensor:
    """Convert integer class index to a one-hot float tensor."""
    target = torch.zeros(NUM_CLASSES)
    target[label] = 1.0
    return target


# --- Dataset WITHOUT target_transform ---
train_data_no_transform = datasets.MNIST(
    root="lessons/section5_pytorch_computer_vision/data",
    train=True,
    download=False,  # already downloaded above
    transform=ToTensor(),
    target_transform=None,
)

# --- Dataset WITH target_transform (one-hot) ---
train_data_one_hot = datasets.MNIST(
    root="lessons/section5_pytorch_computer_vision/data",
    train=True,
    download=False,
    transform=ToTensor(),
    target_transform=one_hot_encode,
)

# --- Side-by-side comparison on the same sample ---
image_raw, label_raw = train_data_no_transform[0]
image_hot, label_hot = train_data_one_hot[0]

print("=== target_transform comparison (same image, index 0) ===")
print(f"Without target_transform: label = {label_raw!r:>5}  | type = {type(label_raw)}")
print(
    f"With    target_transform: label = {label_hot}  | type = {type(label_hot)} | shape = {label_hot.shape}"
)
# Sample output:
# Without target_transform: label =     5  | type = <class 'int'>
# With    target_transform: label = tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.])  | type = <class 'torch.Tensor'> | shape = torch.Size([10])

"""
6. Visualize at least 5 different samples of the MNIST training dataset.
"""
image_0, label_0 = train_data[0]
print(f"Shape of an image in MNIST is {image_0.shape}")
print(f"type of label_0 is {type(label_0)}")
fig = plt.figure(figsize=(9, 9))
row, column = 4, 4

torch.manual_seed(42)
for i in range(1, row * column + 1):
    image_index = torch.randint(low=0, high=len(train_data), size=(1,)).item()
    image_to_plot, label_to_plot = train_data[int(image_index)]
    fig.add_subplot(row, column, i)
    plt.imshow(image_to_plot.squeeze(dim=0), cmap="gray")
    plt.title(class_name[label_to_plot])
    plt.axis(False)

plt.savefig(
    "lessons/section5_pytorch_computer_vision/src/e_line_130_sample_MNIST_figure.png"
)
