import logging
import os

import numpy as np
import ignite.distributed as idist

from torch.utils.data import Dataset, SequentialSampler
from torchvision import datasets
from torchvision import transforms as T
from torchvision.datasets.cifar import CIFAR100
import sys
sys.path.append(os.getcwd())

from augmentations.randaugment import RandAugmentMC
from augmentations.ctaugment import *


logger = logging.getLogger(__name__)

TRANSFORM_CIFAR = {
    'CIFAR10':
        {'mean': (0.4914, 0.4822, 0.4465),  # (0.5, 0.5, 0.5), #
         'std': (0.2471, 0.2435, 0.2616)},  # (0.25, 0.25, 0.25)}, #
    'CIFAR100':
        {'mean': (0.5071, 0.4867, 0.4408),
         'std': (0.2675, 0.2565, 0.2761)
         }
}

STRONG_AUG = {
    'RA': RandAugmentMC(n=2, m=10),
    'CTA': CTAugment()
}


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datapath = self.params.data_dir
        self.name = params.dataset  # dataset name
        self.loader = {}

        assert self.name in ['CIFAR10', 'CIFAR100'], f"[!] Dataset Name is wrong, \
                                                            expected: CIFAR10, CIFAR100  \
                                                            received: {self.name }"
        self.labeled_transform = T.Compose([
            T.RandomHorizontalFlip(),  # random flip with p=0.5
            T.RandomCrop(size=32,
                         # random shift 12.5% TODO: not random yet
                         padding=int(32*0.125),
                         padding_mode='reflect'),
            T.ToTensor(),
            T.Normalize(mean=TRANSFORM_CIFAR[self.name]['mean'],
                        std=TRANSFORM_CIFAR[self.name]['std'],)])

        self.unlabeled_transform = TransformFix(
            mean=TRANSFORM_CIFAR[self.name]['mean'],
            std=TRANSFORM_CIFAR[self.name]['std'],
            strong_aug_method=STRONG_AUG[params.strongaugment])

        self.transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=TRANSFORM_CIFAR[self.name]['mean'],
                        std=TRANSFORM_CIFAR[self.name]['std'],)])
        self.get_dataset()

    def get_dataset(self):
        data_dir = os.path.join(hydra.utils.get_original_cwd(), self.datapath, 'cifar-%s-batches-py' % self.name[5:])
        downloadFlag = not os.path.exists(data_dir)

        try:
            trainset = datasets.__dict__[self.name](data_dir, train=True, download=downloadFlag)
            testset = datasets.__dict__[self.name](data_dir, train=False, transform=self.transform_test, download=False)
        except:
            raise IOError(f'Dataset {self.name} not found in cwd {data_dir}')

        logger.info(f"Dataset: {self.name}")

        self.num_classes = max(testset.targets) + 1

        # sampling labeled and unlabeled data
        labeled_idx, unlabeled_idx = self.sampling(self.params.num_expand_x, trainset)

        # apply transforms
        train_sup_dataset, train_unsup_dataset = self.apply_transform(labeled_idx, unlabeled_idx, trainset)

        self.get_dataloader(train_sup_dataset, train_unsup_dataset, testset)

    def sampling(self, num_expand_x, trainset):
        num_per_class = self.params.label_num // self.num_classes
        labels = np.array(trainset.targets)

        # sample labeled
        categorized_idx = [list(np.where(labels == i)[0]) for i in range(self.num_classes)]
        labeled_idx = [idx for idxs in categorized_idx 
                            for idx in np.random.choice(idxs, num_per_class)]

        # expand the number of labeled to num_expand_x, unlabeled to num_expand_x * 7
        exapand_labeled = num_expand_x // len(labeled_idx)
        exapand_unlabled = num_expand_x * self.params.mu // labels.size

        labeled_idx = labeled_idx * exapand_labeled
        unlabeled_idx = list(np.arange(labels.size)) * exapand_unlabled

        if len(labeled_idx) < num_expand_x:
            diff = num_expand_x - len(labeled_idx)
            labeled_idx.extend(np.random.choice(labeled_idx, diff))
        else:
            assert len(labeled_idx) == num_expand_x

        if len(unlabeled_idx) < num_expand_x * self.params.mu:
            diff = num_expand_x * self.params.mu - len(unlabeled_idx)
            unlabeled_idx.extend(np.random.choice(unlabeled_idx, diff))
        else:
            assert len(unlabeled_idx) == num_expand_x * self.params.mu

        logger.info(f"Labeled examples: {len(labeled_idx)}"
                    f" Unlabeled examples: {len(unlabeled_idx)}")

        return labeled_idx, unlabeled_idx

    def apply_transform(self, labeled_idx, unlabeled_idx, trainset):
        # train_sup_dataset[0]: img, target
        train_sup_dataset = TransformedDataset(
            trainset, labeled_idx,
            transform=self.labeled_transform)

        # train_unsup_dataset[0]: [weak, strong], target
        train_unsup_dataset = TransformedDataset(
            trainset, unlabeled_idx,
            transform=self.unlabeled_transform) 
        return train_sup_dataset, train_unsup_dataset

    def get_dataloader(self, train_sup_dataset, train_unsup_dataset, testset):
        self.loader['labeled'] = idist.auto_dataloader(
            train_sup_dataset,
            # sampler=train_sampler(labeled_dataset),
            shuffle=True,
            pin_memory='cuda' in idist.device().type,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
            drop_last=True)

        self.loader['unlabeled'] = idist.auto_dataloader(
            train_unsup_dataset,
            # sampler=train_sampler(unlabeled_dataset),
            shuffle=True,
            pin_memory='cuda' in idist.device().type,
            batch_size=self.params.batch_size*self.params.mu,
            num_workers=self.params.num_workers,
            drop_last=True)

        self.loader['test'] = idist.auto_dataloader(
            testset,
            pin_memory='cuda' in idist.device().type,
            sampler=SequentialSampler(testset),
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers)


class TransformFix(object):
    def __init__(self, mean, std, strong_aug_method):
        self.weak = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32,
                         padding=int(32*0.125),
                         padding_mode='reflect')])
        self.strong = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32,
                         padding=int(32*0.125),
                         padding_mode='reflect'),
            strong_aug_method])
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.15), value=127), ]  # cutout
        )

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class TransformedDataset(Dataset):
    def __init__(self, dataset, indexs, transform, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        if indexs is not None:
            self.dataset.data = dataset.data[indexs]
            self.dataset.targets = np.array(self.dataset.targets)[indexs]

    def __getitem__(self, i):
        img, target = self.dataset[i]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig
    import matplotlib.pyplot as plt
    @hydra.main(config_path='../config', config_name='config')
    def main(cfg: DictConfig) -> None:
        print(f"Current working directory : {os.getcwd()}")
        print(f"Orig working directory    : {hydra.utils.get_original_cwd()}")
        data = LoadDataset(cfg.DATASET)
        for i, (name, loader) in enumerate(data.loader.items()):
            img_batch, _ = next(iter(loader))
            if isinstance(img_batch, list):
                n = np.random.randint(0, img_batch[0].shape[0])
                im = [T.ToPILImage()(img[n]).convert('RGB')
                      for img in img_batch]
                plt.figure(i)
                plt.subplot(121)
                plt.imshow(im[0])
                plt.title('weakly augmented')
                plt.subplot(122)
                plt.imshow(im[1])
                plt.title('strongly augmented (%s)' %
                          cfg.DATASET.strongaugment)
            else:
                img = img_batch[np.random.randint(0, img_batch.shape[0])]
                img = T.ToPILImage()(img).convert('RGB')
                plt.figure(i)
                plt.imshow(img)
                plt.title(name)

        plt.show()

    main()
