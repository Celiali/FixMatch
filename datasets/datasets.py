import numpy as np
from PIL import Image
import os
import yaml
import argparse
from easydict import EasyDict as edict

import torch
from torch.utils.data import Dataset

import torchvision
from torchvision import transforms

from augmentations import randaugment
import logging

TRANSFORM_CIFAR = {
    'CIFAR10':
        {'mean': (0.4914, 0.4822, 0.4465),
         'std': (0.2471, 0.2435, 0.2616)},
    'CIFAR100':
        {'mean': (0.5071, 0.4867, 0.4408),
         'std': (0.2675, 0.2565, 0.2761)
         }
}

logger = logging.getLogger(__name__)

class LoadDataset_Vanilla(object):
    def __init__(self,params):
        self.params = params
        self.datapath = self.params.data_dir

    def get_dataset(self):
        labeled_transform = transforms.Compose([
            # you can add other transformations in this list
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=TRANSFORM_CIFAR[self.params.dataset ]['mean'],
                                 std=TRANSFORM_CIFAR[self.params.dataset ]['std'], )
        ])

        unlabeled_transform = transforms.Compose([
            # you can add other transformations in this list
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=TRANSFORM_CIFAR[self.params.dataset ]['mean'],
                                 std=TRANSFORM_CIFAR[self.params.dataset ]['std'] )
        ])

        transform = transforms.Compose([
            # you can add other transformations in this list
            transforms.ToTensor(),
            transforms.Normalize(mean=TRANSFORM_CIFAR[self.params.dataset]['mean'],
                                 std=TRANSFORM_CIFAR[self.params.dataset]['std'])
        ])
        if self.params.dataset == 'CIFAR10':
            downloadFlag = not os.path.exists(os.path.join(self.datapath,'cifar-10-batches-py'))
            trainset = torchvision.datasets.CIFAR10(root = self.datapath, train=True,transform=transform, download=downloadFlag)
            testset = torchvision.datasets.CIFAR10(root = self.datapath, train=False, transform=transform, download=downloadFlag)
        elif self.params.dataset == 'CIFAR100':
            downloadFlag = not os.path.exists(os.path.join(self.datapath, 'cifar-100-batches-py'))
            trainset = torchvision.datasets.CIFAR100(root=self.datapath, train=True, transform=transform,download=downloadFlag)
            testset = torchvision.datasets.CIFAR100(root=self.datapath, train=False, transform=transform,download=downloadFlag)

        logger.info(f"[Dataset] Loading original dataset {self.params.dataset}")
        logger.info(f"Training examples: {len(trainset)}"
                    f" Testing examples: {len(testset)}")

        return trainset,None,testset


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='Training code')
    parser.add_argument('--config', default='../config/config.yaml', type=str, help='yaml config file')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        try:
            config_file = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    CONFIG = edict(config_file)
    print('==> CONFIG is: \n', CONFIG, '\n')

    data = LoadDataset_Vanilla(CONFIG.DATASET)
    trainset, _, testset = data.get_dataset()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                              shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=4)
    print(len(trainset)) #print(len(trainloader))

    batchdata = next(iter(trainloader))
    print(batchdata[0])
    print(batchdata[1])

    x = batchdata[0]
    import matplotlib.pyplot as plt
    a = x[0][:]
    plt.imshow(a.permute(1,2,0))
    plt.show()
    # plt.imsave('./test.png',a.permute(1,2,0).numpy())
    # # add mean and std back to the image
    # t = a.permute(1,2,0).numpy()
    # t2 = t* TRANSFORM_CIFAR[CONFIG.DATASET.dataset ]['std']+TRANSFORM_CIFAR[CONFIG.DATASET.dataset ]['mean']
    # plt.imshow(t2)
    # plt.show()
