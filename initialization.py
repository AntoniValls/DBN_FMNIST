# -*- coding: utf-8 -*-
"""
Firstly, we download some scripts implementing a Deep Belief Network and a Restricted Boltzmann Machine in PyTorch. We also import the necessary libraries.
"""

def get_dbn_library():
  files = ["DBN.py", "RBM.py"]
  repository_url = "https://raw.githubusercontent.com/flavio2018/Deep-Belief-Network-pytorch/master/"
  for file in files:
    ! wget -O {file} {repository_url}{file}

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# get_dbn_library()

import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import scipy.cluster as cluster
import sklearn.preprocessing
import torch
import torch.nn.functional as functional
import os
import random
import torchvision as tv
from tqdm.notebook import tqdm

from DBN import DBN

"""We choose dynamically the kind of device used for computations (CPU or GPU)."""

print(torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

"""We set a seed for repeteability."""

def set_seed(seed, use_gpu = True):
    """
    Set SEED for PyTorch reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_gpu:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

SEED = 3543

USE_SEED = True

if USE_SEED:
    set_seed(SEED, torch.cuda.is_available())

"""Secondly, we download the FashionMNIST (both the train and test datasets). The FashionMNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes:
  * T-shirt/top
  * Trouser
  * Pullover
  * Dress
  * Coat
  * Sandal
  * Shirt
  * Sneaker
  * Bag
  * Ankle boot

This dataset can be load directly from TorchVision.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# # Load the dataset
# fmnist_tr = tv.datasets.FashionMNIST(root="../fmnist", train=True, download=True)
# fmnist_te = tv.datasets.FashionMNIST(root="../fmnist", train=False, download=True)

print("-------  DATASET  INFO  --------")
print(fmnist_tr)
print()
print(fmnist_tr.data.shape)

"""Let's examin the image tensors and the distribution of the FashionMNIST dataset."""

# Print shape, min and max grayscale values from one image

idx = 67

print(f'Image shape: {fmnist_tr.data[idx].shape}\n')
min_value = fmnist_tr.data[idx].min()
print(f"Minimum grayscale values : {min_value}")
max_value =fmnist_tr.data[idx].max()
print(f"Maximum grayscale values : {max_value}")

# Check the distribution of classes in both training and test dataset

tr_counts = [0] * 10
te_counts = [0] * 10

for img, label in fmnist_tr:
    tr_counts[label] += 1

for img, label in fmnist_te:
    te_counts[label] += 1

print("Training Dataset:")
for i, cls in enumerate(fmnist_tr.classes):
    print(f"  {cls}: {tr_counts[i]}")

print("\nTest Dataset:")
for i, cls in enumerate(fmnist_te.classes):
    print(f"  {cls}: {te_counts[i]}")

"""As we can see, the distribution of classes is uniform.

We need to preprocess the images in order to make them suitable for the DBN model we will use. We divide both subsets by 255 in order to have each pixel values in the [0,1] range. We also send the data to CUDA.
"""

fmnist_tr.data = fmnist_tr.data / 255
fmnist_te.data = fmnist_te.data / 255

fmnist_tr.data = fmnist_tr.data.to(device)
fmnist_te.data = fmnist_te.data.to(device)
fmnist_tr.targets = fmnist_tr.targets.to(device)
fmnist_te.targets = fmnist_te.targets.to(device)

"""Let's visualize one image of each class from the training set."""

fig, axs = plt.subplots(1, len(fmnist_tr.classes), figsize=(20, 20))

for i, cls in enumerate(fmnist_tr.classes):
    for j, img in enumerate(fmnist_tr.data.cpu()):
        if fmnist_tr.targets[j].cpu() == i:
            axs[i].imshow(img, cmap='gray')
            axs[i].set_title(cls)
            axs[i].axis('off')
            break

plt.show()
