import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2

from common.device import get_best_device, print_device_info

from typing import Tuple, Dict, List
from pathlib import Path
import pathlib
import os
import random
from PIL import Image
import matplotlib.pyplot as plt

"""
0. Setting up device-agnostic code
"""
device = get_best_device()
print_device_info(device)

"""
1. Get data

Dataset will be used here is a subset of Food101 dataset
3 classes and 10% of the images

When starting out ML project, it's important to try things on a small scale then increase
when necessary

Note: we are having standard image class structure
"""
# Set up path to a data folder
data_path = Path("lessons/section6_pytorch_custom_datasets/data")
image_path = data_path / "pizza_steak_sushi"

# Setup training and testing path
train_dir = image_path / "train"
test_dir = image_path / "test"

# Write a transform for image (using torchvision.transforms.v2)
data_transform = v2.Compose(
    [
        v2.Resize(
            (64, 64)
        ),  # resize all images to 64x64, this is a hyperparameter you can change
        v2.RandomHorizontalFlip(
            p=0.5
        ),  # data augmentation, randomly flip some images horizontally
        v2.ToImage(),  # PIL Image -> uint8 tensor [C, H, W]
        v2.ToDtype(torch.float32, scale=True),  # uint8 [0, 255] -> float32 [0.0, 1.0]
    ]
)

"""
5. Option 2: Loading Image data with a custom dataset class
This is the most flexible option but also requires the most code to set up and performance 
may not be as good as using a built-in dataset class like ImageFolder

1. Want to be able to load image from file
2. Get class names from dataset
3. Get classes as dictionary from dataset
"""

"""
5.1 Creating a helper to get class names

Steps:
1. Get the class names using os.scandir() to scan the training directory for subdirectories (each subdirectory represents a class).
2. Raise an error if no class names are found (i.e., if there are no subdirectories in the training directory).
3. Turn the class names into a list and dictionary and return them.
"""

# Setup target directory
target_directory = train_dir
print(f"Target directory: {target_directory}")

# Get the class names from the target directory
class_names_found = sorted(
    [entry.name for entry in os.scandir(target_directory) if entry.is_dir()]
)
print(f"Class names found: {class_names_found}")


def find_classes(directory: str | Path) -> Tuple[List[str], Dict[str, int]]:
    """
    Finds the class folders in a dataset.

    Args:
        directory (str | Path): The root directory of the dataset.

    Returns:
        Tuple[List[str], Dict[str, int]]: A tuple containing a list of class names and a dictionary mapping class names to indices.
    """
    # Get the class names from the target directory
    class_names_found = sorted(
        [entry.name for entry in os.scandir(directory) if entry.is_dir()]
    )

    # Raise an error if no class names are found
    if not class_names_found:
        raise FileNotFoundError(f"Couldn't find any class folders in {directory}.")

    # Create a mapping of class names to indices
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names_found)}

    return class_names_found, class_to_idx


class_names, class_to_idx = find_classes(target_directory)
print(f"Class names: {class_names} | Class to index mapping: {class_to_idx}")

"""
5.2 Creating a custom dataset class to replicate ImageFolder (for exercise purpose, in practice you would just use ImageFolder)

Steps:
1. Subclass torch.utils.data.Dataset to create a custom dataset class.
2. Init our subclass with a target directory and a transform.
3. Create several attributes in the init method (some samples):
    a. paths - path to all the images in the dataset
    b. transform - the transform to apply to the images
    c. class_names - the class names found in the dataset
    d. class_to_idx - the class name to index mapping
4. Create a function to load_images() which will use os.walk() to walk through the target directory and get the paths to all the images in the dataset.
5. Override the __len__ method to return the number of samples in the dataset.
6. Override the __getitem__ method to load an image and its label given an index.
"""


# 0. Write a custom dataset class to load the image data (above import)
# 1. Subclass torch.utils.data.Dataset to create a custom dataset class.
class ImageFolderCustom(Dataset):
    # 2. Init our subclass with a target directory and a transform.
    def __init__(self, target_dir: str | Path, transform: v2.Compose = None) -> None:
        # 3. Create several attributes in the init method (some samples):
        self.paths = list(
            pathlib.Path(target_dir).rglob("*/*.jpg")
        )  # a. paths - path to all the images in the dataset
        # Setup transforms
        self.transform = (
            transform  # b. transform - the transform to apply to the images
        )
        # Get class names and class to index mapping using the helper function we created
        self.class_names, self.class_to_idx = find_classes(
            target_dir
        )  # c. class_names - the class names found in the dataset, d. class_to_idx - the class name to index mapping

    # 4. Create a function to load_images() which will use os.walk() to walk through the target directory and get the paths to all the images in the dataset.
    def load_images(self, index: int) -> Image.Image:
        """
        Load an image given an index.

        Args:
            index (int): The index of the image to load.

        Returns:
            Image.Image: The loaded image.
        """
        image_path = self.paths[index]
        image = Image.open(image_path)
        return image

    # 5. Override the __len__ method to return the number of samples in the dataset.
    def __len__(self) -> int:
        return len(self.paths)

    # 6. Override the __getitem__ method to load an image and its label given an index.
    def __getitem__(self, index: int) -> Tuple[torch.Tensor | Image.Image, int]:
        """
        Load an image and its label given an index.

        Args:
            index (int): The index of the image to load.

        Returns:
            Tuple[torch.Tensor|Image.Image, int]: The loaded image and its label.
        """
        image = self.load_images(index)
        # get the class name from the parent directory of the image
        class_name = self.paths[index].parent.name

        # get the class index from the class name
        class_idx = self.class_to_idx[class_name]

        # transform the image if a transform is provided
        if self.transform:
            image = self.transform(image)
        return image, class_idx


# Create a transform first (using torchvision.transforms.v2)
train_transform = v2.Compose(
    [
        v2.Resize(size=(64, 64)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(),  # PIL Image → uint8 tensor [C, H, W]
        v2.ToDtype(torch.float32, scale=True),  # uint8 [0, 255] → float32 [0.0, 1.0]
    ]
)

test_transform = v2.Compose(
    [
        v2.Resize(size=(64, 64)),
        v2.ToImage(),  # PIL Image → uint8 tensor [C, H, W]
        v2.ToDtype(torch.float32, scale=True),  # uint8 [0, 255] → float32 [0.0, 1.0]
    ]
)

# Test out ImageFolderCustom dataset class
train_data_custom = ImageFolderCustom(target_dir=train_dir, transform=train_transform)
test_data_custom = ImageFolderCustom(target_dir=test_dir, transform=test_transform)
print(f"\vNumber of training samples: {len(train_data_custom)}")
print(f"Number of testing samples: {len(test_data_custom)}")
print(f"Class names: {train_data_custom.class_names}")
print(f"Class to index mapping: {train_data_custom.class_to_idx}")

# Check if the ImageFolderCustom dataset class works the same as ImageFolder
train_data_imagefolder = datasets.ImageFolder(root=train_dir, transform=train_transform)
print(
    f"\nClass name comparison: {train_data_custom.class_names == train_data_imagefolder.classes}"
)
print(
    f"Class to index mapping comparison: {train_data_custom.class_to_idx == train_data_imagefolder.class_to_idx}"
)

"""
5.3 Create a function to visualize random images from the custom dataset class

Steps:
1. Take in a Dataset, number of images to show and number of images want to show.
2. Cap the number of images at 10
3. Set the random seed
4. Get a list of random samples from target dataset
5. Set up a matplotlib figure and axes
6. Loop through the random samples and plot the original and transformed images side by side with their
7. Make the dimension of image is good for visualization in matplotlib
"""


# 1. Take in a Dataset, number of images to show and number of images want to show.
def display_random_images_from_dataset(
    dataset: Dataset,
    class_name: List[str],
    n: int = 10,
    display_shape: bool = True,
    seed: int = 42,
) -> None:
    # 2. Cap the number of images at 10
    n = min(n, 10)
    # 3. Set the random seed
    if seed is not None:
        random.seed(seed)

    # 4. Get a list of random samples from target dataset
    indices = random.sample(range(len(dataset)), k=n)

    # 5. Set up a matplotlib figure with enough width for n images
    plt.figure(figsize=(n * 3, 4))

    # 6. Loop through the random samples and plot the original and transformed images side by side with their class names as title
    for i, target_sample in enumerate(indices):
        image, label = dataset[target_sample]
        class_name_found = class_name[label]
        plt.subplot(1, n, i + 1)

        # [C, H, W] -> [H, W, C] for matplotlib
        plt.imshow(image.permute(1, 2, 0))

        title = f"Class:\n{class_name_found}"
        if display_shape:
            title += f"\n{image.shape}"
        plt.title(title, fontsize=8)
        plt.axis(False)
    plt.tight_layout()


# Display random images from the ImageFolder dataset
display_random_images_from_dataset(
    train_data_imagefolder, class_name=class_names, n=10, seed=42
)
plt.savefig(
    "lessons/section6_pytorch_custom_datasets/src/a_line_285_random_images_from_imagefolder.png"
)

# Using the custom dataset class to display random images
display_random_images_from_dataset(
    train_data_custom, class_name=class_names, n=10, seed=42
)
plt.savefig(
    "lessons/section6_pytorch_custom_datasets/src/a_line_293_random_images_from_custom_dataset.png"
)

"""
5.4 Turn the custom dataset into a DataLoader for training and testing
"""
BATCH_SIZE = 32
train_dataloader_custom = DataLoader(
    dataset=train_data_custom,
    batch_size=BATCH_SIZE,
    num_workers=1,
    shuffle=True,  # shuffle the data for training
)
test_dataloader_custom = DataLoader(
    dataset=test_data_custom,
    batch_size=BATCH_SIZE,
    num_workers=1,
    shuffle=False,  # no need to shuffle test data, it will not be used in training
)

# Check the dataloader
print(f"Number of batches in train dataloader: {len(train_dataloader_custom)}")
print(f"Number of batches in test dataloader: {len(test_dataloader_custom)}")

# Get image and label from dataloader
image_batch, label_batch = next(iter(train_dataloader_custom))
print(
    f"Image batch shape: {image_batch.shape} | Label batch shape: {label_batch.shape}"
)


"""
6. Other forms of transforms (data augmentation and normalization)

Looks at one type of augmentation to train PyTorch vision model
"""
# Look at trivial augmentation
train_trivial_augmentation = v2.Compose(
    [
        v2.Resize(size=(224, 224)),
        v2.TrivialAugmentWide(
            num_magnitude_bins=5
        ),  # apply a random augmentation to the image (from a list of 5 augmentations)
        v2.ToImage(),  # PIL Image → uint8 tensor [C, H, W]
        v2.ToDtype(torch.float32, scale=True),  # uint8 [0, 255] → float32 [0.0, 1.0]
    ]
)

test_trivial_augmentation = v2.Compose(
    [
        v2.Resize(size=(224, 224)),
        v2.ToImage(),  # PIL Image → uint8 tensor [C, H, W]
        v2.ToDtype(torch.float32, scale=True),  # uint8 [0, 255] → float32 [0.0, 1.0]
    ]
)

# Get all image paths
image_paths_list = list(image_path.glob("*/*/*.jpg"))
print(f"Total number of images: {len(image_paths_list)}")


# Plot random transformed images
def plot_transformed_image(image_paths, transform, n=3, seed=None):
    """
    Plot n random images in a single figure with 2 columns:
      - Left column: original image
      - Right column: transformed image
    """
    if seed:
        random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)

    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(8, n * 3))

    for row, image_path in enumerate(random_image_paths):
        with Image.open(image_path) as f:
            class_name = image_path.parent.stem

            # Left: original image
            axes[row, 0].imshow(f)
            axes[row, 0].set_title(f"Original | {class_name}\nSize: {f.size}")
            axes[row, 0].axis(False)

            # Right: transformed image
            # transform returns [C, H, W] tensor; permute to [H, W, C] for matplotlib
            transformed_image = transform(f)
            assert isinstance(transformed_image, torch.Tensor)
            axes[row, 1].imshow(transformed_image.permute(1, 2, 0))
            axes[row, 1].set_title(
                f"Transformed | {class_name}\nShape: {transformed_image.shape}"
            )
            axes[row, 1].axis(False)


plot_transformed_image(
    image_paths=image_paths_list, transform=train_trivial_augmentation, n=3, seed=42
)
plt.tight_layout()
plt.savefig(
    "lessons/section6_pytorch_custom_datasets/src/a_line_385_trivial_augmentation_transformed_images.png"
)
