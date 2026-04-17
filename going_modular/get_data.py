"""
Not part of the course but a helper script to download and prepare the data for the course.
run hancheng_dl_python going_modular/get_data.py
"""

import os
import zipfile

from pathlib import Path

import requests

# Setup path to data folder
data_path = Path("going_modular/data/")
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / "train"
test_dir = image_path / "test"

# If the image folder doesn't exist, download it and prepare it...
if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

# Download pizza, steak, sushi data
with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
    request = requests.get(
        "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
    )
    print("Downloading pizza, steak, sushi data...")
    f.write(request.content)

# Unzip pizza, steak, sushi data
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("Unzipping pizza, steak, sushi data...")
    zip_ref.extractall(image_path)

# Remove zip file
os.remove(data_path / "pizza_steak_sushi.zip")

"""
Exercise 1 for section 7
Turn the code to get the data (from section 1. Get Data above) into a Python script, such as get_data.py.
When you run the script using python get_data.py it should check if the data already exists and skip downloading if it does.
If the data download is successful, you should be able to access the pizza_steak_sushi images from the data directory.

This is already done in the code above, you can run it to download the data and prepare it for the rest of the course.
"""
