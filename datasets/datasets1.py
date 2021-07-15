import logging
import os
import sys
from collections import defaultdict

from torch.utils.data.sampler import Sampler

sys.path.append(os.getcwd())
from PIL import Image
import torch
import numpy as np
import ignite.distributed as idist
import hydra
import copy
import json
import itertools
from torch.utils.data import Dataset, SequentialSampler
from torch._six import int_classes as _int_classes
from torchvision import datasets
from torchvision import transforms as T
from torchvision.datasets.cifar import CIFAR100
from augmentations.randaugment import RandAugment, CutoutAbs
from augmentations.ctaugment import *

# from test_dataloader import *

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
    'RA': RandAugment(n=2, m=10),
    'CTA': CTAugment()
}


class CombineDataset(Dataset):
    def __init__(self, datasets_to_combine):
        super(CombineDataset, self).__init__()

        self.transform = datasets_to_combine[0].transform
        self.target_transform = datasets_to_combine[0].target_transform
        self.targets = list(itertools.chain(*[dt.targets for dt in datasets_to_combine]))
        self.data = list(itertools.chain(*[dt.data for dt in datasets_to_combine]))

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.data)


class SubsetDataset(Dataset):
    def __init__(self, subset, transform, target_transform):
        self.subset = subset
        self.targets = []
        self.data = []

        for x, y in subset:
            self.targets.append(x)
            self.data.append(y)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.subset)


class KFoldDataset:
    def __init__(self, dataset_to_kfold, k):
        super(KFoldDataset, self).__init__()
        self.transform = dataset_to_kfold.transform
        self.target_transform = dataset_to_kfold.target_transform

        num_dataset = len(dataset_to_kfold)
        num_each_fold = num_dataset // k
        leftover = num_dataset - num_each_fold * k

        self.folds = torch.utils.data.random_split(dataset_to_kfold, [num_each_fold] * (k - 1) +
                                                   [num_each_fold + leftover])
        self.folds = [SubsetDataset(fld_subset, self.transform, self.target_transform) for fld_subset in self.folds]


class LoadDataset_Label_Unlabel(object):
    def __init__(self, params):
        self.params = params
        self.datapath = self.params.data_dir
        self.name = params.dataset  # dataset name
        self.strongaugment = STRONG_AUG[params.strongaugment]
        self.loader = {}

        assert self.name in ['CIFAR10', 'CIFAR100'], f"[!] Dataset Name is wrong, \
                                                            expected: CIFAR10, CIFAR100  \
                                                            received: {self.name}"
        self.labeled_transform = T.Compose([
            T.RandomApply([
                T.RandomCrop(size=32,
                             padding=int(32 * 0.125),
                             padding_mode='reflect'),
            ], p=0.5),
            T.RandomHorizontalFlip(),  # random flip with p=0.5
            T.ToTensor(),
            T.Normalize(mean=TRANSFORM_CIFAR[self.name]['mean'],
                        std=TRANSFORM_CIFAR[self.name]['std'], )])
        ########## Correction ##########
        ########## paper lack of clarification ##########
        # The paper didn't clearly show the probability of translation, I set p=0.5 here

        self.unlabeled_transform = TransformFix(
            mean=TRANSFORM_CIFAR[self.name]['mean'],
            std=TRANSFORM_CIFAR[self.name]['std'],
            strong_aug_method=self.strongaugment,
            both_strong=self.params.both_strong)

        self.transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=TRANSFORM_CIFAR[self.name]['mean'],
                        std=TRANSFORM_CIFAR[self.name]['std'], )])
        # self.get_dataset()

    def get_cta(self):
        return self.strongaugment

    def vanilla_dataset_to_unlabeled(self, dataset):
        self.num_classes = max(dataset.targets) + 1

        # sampling labeled and unlabeled data
        labeled_idx, unlabeled_idx, valid_idx = self.sampling(self.params.num_expand_x, dataset)

        # apply transforms
        labeledSet, unlabeledSet, valid_dataset = self.apply_transform(labeled_idx, unlabeled_idx, valid_idx, dataset)

        if self.params.add_noisy_label:
            labeledSet.dataset = copy.deepcopy(labeledSet.dataset)  # to keep unlabledSet unchange
            unique_idx_labeled = set(labeledSet.indexs)
            unique_idx_valid = set(valid_dataset.indexs)

            valid_idx_cls0 = [i for i in unique_idx_valid if valid_dataset.dataset.targets[i] == 0]
            valid_idx_cls1 = [i for i in unique_idx_valid if valid_dataset.dataset.targets[i] == 1]

            sampled_idx_cls2 = [i for i in unique_idx_labeled if labeledSet.dataset.targets[i] == 2]
            sampled_idx_cls3 = [i for i in unique_idx_labeled if labeledSet.dataset.targets[i] == 3]

            # replacing 3 images
            labeledSet.dataset.data[sampled_idx_cls2[:3]] = valid_dataset.dataset.data[valid_idx_cls0[:3]]
            labeledSet.dataset.data[sampled_idx_cls3[:3]] = valid_dataset.dataset.data[valid_idx_cls1[:3]]

            # removing the three indexs
            for i in range(3):
                valid_dataset.indexs.remove(valid_idx_cls0[i])
                valid_dataset.indexs.remove(valid_idx_cls1[i])
        if isinstance(self.strongaugment, CTAugment):
            ctaDataset = TransformedDataset(dataset, labeled_idx, transform=self.cta_probe_transform)
            return labeledSet, unlabeledSet, valid_dataset, ctaDataset
        return labeledSet, unlabeledSet, valid_dataset

    def get_vanila_dataset(self):
        rootdir = hydra.utils.get_original_cwd()
        data_dir = os.path.join(rootdir, self.datapath, 'cifar-%s-batches-py' % self.name[5:])
        downloadFlag = not os.path.exists(data_dir)

        try:
            trainset = datasets.__dict__[self.name](data_dir, train=True, download=downloadFlag)
            testset = datasets.__dict__[self.name](data_dir, train=False, transform=self.transform_test, download=False)
        except:
            raise IOError(f'Dataset {self.name} not found in cwd {data_dir}')

        logger.info(f"Dataset: {self.name}")
        return trainset, testset

    def get_dataset(self):
        rootdir = hydra.utils.get_original_cwd()
        data_dir = os.path.join(rootdir, self.datapath, 'cifar-%s-batches-py' % self.name[5:])
        downloadFlag = not os.path.exists(data_dir)

        try:
            trainset = datasets.__dict__[self.name](data_dir, train=True, download=downloadFlag)
            testset = datasets.__dict__[self.name](data_dir, train=False, transform=self.transform_test, download=False)
        except:
            raise IOError(f'Dataset {self.name} not found in cwd {data_dir}')

        logger.info(f"Dataset: {self.name}")
        self.num_classes = max(testset.targets) + 1

        # sampling labeled and unlabeled data
        labeled_idx, unlabeled_idx, valid_idx = self.sampling(self.params.num_expand_x, trainset)

        # apply transforms
        labeledSet, unlabeledSet, valid_dataset = self.apply_transform(labeled_idx, unlabeled_idx, valid_idx, trainset)

        if self.params.add_noisy_label:
            labeledSet.dataset = copy.deepcopy(labeledSet.dataset)  # to keep unlabledSet unchange
            unique_idx_labeled = set(labeledSet.indexs)
            unique_idx_valid = set(valid_dataset.indexs)

            valid_idx_cls0 = [i for i in unique_idx_valid if valid_dataset.dataset.targets[i] == 0]
            valid_idx_cls1 = [i for i in unique_idx_valid if valid_dataset.dataset.targets[i] == 1]

            sampled_idx_cls2 = [i for i in unique_idx_labeled if labeledSet.dataset.targets[i] == 2]
            sampled_idx_cls3 = [i for i in unique_idx_labeled if labeledSet.dataset.targets[i] == 3]

            # replacing 3 images
            labeledSet.dataset.data[sampled_idx_cls2[:3]] = valid_dataset.dataset.data[valid_idx_cls0[:3]]
            labeledSet.dataset.data[sampled_idx_cls3[:3]] = valid_dataset.dataset.data[valid_idx_cls1[:3]]

            # removing the three indexs
            for i in range(3):
                valid_dataset.indexs.remove(valid_idx_cls0[i])
                valid_dataset.indexs.remove(valid_idx_cls1[i])
        if isinstance(self.strongaugment, CTAugment):
            ctaDataset = TransformedDataset(trainset, labeled_idx, transform=self.cta_probe_transform)
            return labeledSet, unlabeledSet, valid_dataset, testset, ctaDataset
        return labeledSet, unlabeledSet, valid_dataset, testset

        # self.get_dataloader(train_sup_dataset, train_unsup_dataset, testset)

    def cta_probe_transform(self, img):
        policy = self.strongaugment.get_policy(probe=True)
        probe = self.strongaugment.apply(img, policy)
        probe = self.transform_test(CutoutAbs(probe, 16))
        return probe, json.dumps(policy)

    def get_labeled_valid(self, cat_idx, num_per_class, valid_per_class=500):
        valid_idx = []
        labeled_idx = []
        for idxs in cat_idx:
            idx = np.random.choice(idxs, num_per_class + valid_per_class, replace=False)
            labeled_idx = np.concatenate((labeled_idx, idx[:num_per_class]), axis=None)
            valid_idx = np.concatenate((valid_idx, idx[num_per_class:]), axis=None)
        # the default value is "replace = True" for np.random.choice, but we don't want to sample the same image twice
        # in labeled data. Thus, I add replace = False.
        return list(labeled_idx.astype(int)), list(valid_idx.astype(int))

    def get_labeled_valid_barely(self, cat_idx, num_per_class, valid_per_class=500):
        valid_idx = []
        labeled_idx = []

        selected_idx = [4255, 6446, 8580, 11759, 12598, 29349, 29433, 33759, 35345, 38639]
        for idxs in cat_idx:
            for j in range(len(selected_idx)):
                s_idx = selected_idx[j]
                if s_idx in idxs:
                    idxs.remove(s_idx)
                    break
            idx = np.random.choice(idxs, valid_per_class, replace=False)
            labeled_idx = np.concatenate((labeled_idx, s_idx), axis=None)
            valid_idx = np.concatenate((valid_idx, idx), axis=None)
        # the default value is "replace = True" for np.random.choice, but we don't want to sample the same image twice
        # in labeled data. Thus, I add replace = False.
        return list(labeled_idx.astype(int)), list(valid_idx.astype(int))

    def sampling(self, num_expand_x, trainset):  # num_expand_x: 2^16 #expected total number of labeled training samples
        num_per_class = self.params.label_num // self.num_classes
        labels = np.array(trainset.targets)

        # sample labeled
        categorized_idx = [list(np.where(labels == i)[0]) for i in range(self.num_classes)]  # [[], [],]

        # Get labeled data and validation data index
        # The type of labeled_idx should be "list"
        if self.params.label_num == 10 and self.params.barely:
            labeled_idx, valid_idx = self.get_labeled_valid_barely(categorized_idx, num_per_class)
        else:
            labeled_idx, valid_idx = self.get_labeled_valid(categorized_idx, num_per_class)

        # Update the training data index since we will not use validation set
        unlabeled_idx = np.array(np.setdiff1d(np.arange(labels.size), np.array(valid_idx)))

        # expand the number of labeled to num_expand_x, unlabeled to num_expand_x * 7
        exapand_labeled = num_expand_x // len(labeled_idx)  # len(labeled_idx) = 40 00
        exapand_unlabled = num_expand_x * self.params.mu // unlabeled_idx.size  # labels.size

        # labels.size = 50 000, all the samples are used in the unlabel data set

        labeled_idx = labeled_idx * exapand_labeled
        unlabeled_idx = list(unlabeled_idx) * exapand_unlabled  #

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
                    f" Unlabeled examples: {len(unlabeled_idx)}"
                    f"Validation examples: {len(valid_idx)}"
                    )

        return labeled_idx, unlabeled_idx, valid_idx

    def apply_transform(self, labeled_idx, unlabeled_idx, valid_idx, trainset):
        # train_sup_dataset[0]: img, target
        train_sup_dataset = TransformedDataset(
            trainset, labeled_idx,
            transform=self.labeled_transform)

        # train_unsup_dataset[0]: [weak, strong], target
        train_unsup_dataset = TransformedDataset(
            trainset, unlabeled_idx,
            transform=self.unlabeled_transform)

        valid_dataset = TransformedDataset(
            trainset, valid_idx,
            transform=self.transform_test)

        return train_sup_dataset, train_unsup_dataset, valid_dataset

    def get_dataloader(self):
        train_sup_dataset, train_unsup_dataset, testset = self.get_dataset()
        if self.params.batch_balanced:
            kwargs = dict(
                pin_memory='cuda' in idist.device().type,
                num_workers=self.params.num_workers,
            )
        else:
            kwargs = dict(
                pin_memory='cuda' in idist.device().type,
                num_workers=self.params.num_workers,
                shuffle=True,
                batch_size=self.params.batch_size,
            )

        self.loader['labeled'] = idist.auto_dataloader(train_sup_dataset, **kwargs,
                                                       batch_sampler=BatchWeightedRandomSampler(train_sup_dataset,
                                                                                                batch_size=self.params.batch_size) if self.params.batch_balanced else None,

                                                       )

        self.loader['unlabeled'] = idist.auto_dataloader(train_unsup_dataset, **kwargs,
                                                         batch_sampler=BatchWeightedRandomSampler(train_unsup_dataset,
                                                                                                  batch_size=self.params.batch_size) if self.params.batch_balanced else None,
                                                         )

        self.loader['test'] = idist.auto_dataloader(testset, **kwargs, sampler=SequentialSampler(testset))
        return self.loader


class BatchWeightedRandomSampler(Sampler):
    '''Samples elements for a batch with given probabilites of each element'''

    def __init__(self, data_source, batch_size, drop_last=False):
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.targets = np.array(data_source.dataset.targets)[data_source.indexs]
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        nclass = max(self.targets) + 1
        sample_distrib = np.array([len(np.where(self.targets == i)[0]) for i in range(nclass)])
        sample_distrib = sample_distrib / sample_distrib.max()

        class_id = defaultdict(list)
        for idx, c in enumerate(self.targets):
            class_id[c].append(idx)

        assert min(class_id.keys()) == 0 and max(class_id.keys()) == (nclass - 1)
        class_id = [np.array(class_id[i], dtype=np.int64) for i in range(nclass)]

        for i in range(nclass):
            np.random.shuffle(class_id[i])

        # rerange indexs following the rule so that labels are ranged like: 0,1,....9,0,....9,...
        # adopted from https://github.com/google-research/fixmatch/blob/79f9fd3e6267035d685864beaec40dd45408ecb0/scripts/create_split.py#L87
        npos = np.zeros(nclass, np.int64)
        label = []
        for i in range(len(self.targets)):
            c = np.argmax(sample_distrib - npos / max(npos.max(), 1))
            label.append(class_id[c][npos[c]])  # the indexs of examples
            npos[c] += 1

        batch = []
        for idx in label:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class TransformFix(object):
    def __init__(self, mean, std, strong_aug_method, both_strong=False):
        self.strong = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32,
                         padding=int(32 * 0.125),
                         padding_mode='reflect'),
            strong_aug_method])
        if both_strong:
            self.weak = self.strong
        else:
            self.weak = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomCrop(size=32,
                             padding=int(32 * 0.125),
                             padding_mode='reflect')])
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            # T.RandomErasing(scale=(0.02, 0.15), value=127), # cutout
        ]
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
        self.indexs = indexs
        # if indexs is not None:
        #     self.dataset.data = dataset.data[indexs]
        #     self.dataset.targets = np.array(self.dataset.targets)[indexs]

    def __getitem__(self, i):
        img, target = self.dataset[self.indexs[i]]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.indexs)


if __name__ == '__main__':
    from omegaconf import DictConfig
    import matplotlib.pyplot as plt
    import torch


    @hydra.main(config_path='../dataset', config_name='config')
    def main(cfg: DictConfig) -> None:
        print(f"Current working directory : {os.getcwd()}")
        print(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

        def restore_stats(img):
            mean = TRANSFORM_CIFAR[cfg.DATASET.dataset]['mean']
            mean = torch.tensor(mean).unsqueeze(dim=1).unsqueeze(dim=1)
            std = TRANSFORM_CIFAR[cfg.DATASET.dataset]['std']
            std = torch.tensor(std).unsqueeze(dim=1).unsqueeze(dim=1)
            img = img * std + mean
            return T.ToPILImage()(img).convert('RGB')

        def showImg(dataset: TransformedDataset, name, index=None):
            idx = index or np.random.randint(0, len(dataset))
            plt.figure(idx)
            if name == 'test':
                plt.imshow(restore_stats(dataset[0][0]))
                plt.title(name)
            else:
                raw_img, _ = dataset.dataset[dataset.indexs[idx]]
                transformed_img, _ = dataset[idx]  # transformed img

                if isinstance(transformed_img, tuple):
                    im = [restore_stats(img) for img in transformed_img]
                    plt.subplot(131)
                    plt.imshow(raw_img)
                    plt.title('raw image')
                    plt.subplot(132)
                    plt.imshow(im[0])
                    plt.title('weakly augmented')
                    plt.subplot(133)
                    plt.imshow(im[1])
                    plt.title('strongly augmented (%s)' %
                              cfg.DATASET.strongaugment)
                else:
                    transformed_img = restore_stats(transformed_img)
                    plt.subplot(121)
                    plt.imshow(raw_img)
                    plt.title('raw image')
                    plt.subplot(122)
                    plt.imshow(transformed_img)
                    plt.title(name)

        data = LoadDataset_Label_Unlabel(cfg.DATASET)
        dataset = data.get_dataset()
        for name, ds in zip(['train labeled', 'train unlabled', 'test'], dataset):
            showImg(ds, name, index=0)
        plt.show()
        # test_dataloader(data,cfg,TRANSFORM_CIFAR)


    main()
