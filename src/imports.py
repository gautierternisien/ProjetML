import os
from PIL import Image

import numpy as np

import pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt

from PIL import Image

from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import optuna
import warnings
from sklearn.model_selection import cross_val_score
import zipfile
import urllib.request
