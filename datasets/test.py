import numpy as np
from torch.utils.data.dataset import Subset
from torchvision import datasets
import matplotlib.pyplot as plt

dataset = datasets.CIFAR10('./data/cifar-10-batches-py', train=True, download=False)

indexs = np.loadtxt('./datasets/prototypical.txt')

dataset = Subset(dataset, indexs)

