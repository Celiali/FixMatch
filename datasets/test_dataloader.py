from torchvision import utils
import matplotlib.pyplot as plt
import torch
from torchvision import transforms as T


def restore_stats(img,cfg,TRANSFORM_CIFAR):
    mean = TRANSFORM_CIFAR[cfg.DATASET.dataset]['mean']
    mean = torch.tensor(mean).unsqueeze(dim=1).unsqueeze(dim=1)
    std = TRANSFORM_CIFAR[cfg.DATASET.dataset]['std']
    std = torch.tensor(std).unsqueeze(dim=1).unsqueeze(dim=1)
    img = img * std + mean
    return T.ToPILImage()(img).convert('RGB')

def test_dataloader(data, cfg, TRANSFORM_CIFAR):
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    loader = data.get_dataloader()
    lb_tfm_dataloader = loader['labeled']
    un_tfm_dataloader = loader['unlabeled']
    tst_dataloader = loader['test']

    # DATA LOADER TEST FOR UNLABELED DATA
    for i_batch, sample_batched in enumerate(un_tfm_dataloader):
        images, targets = sample_batched
        # images[0]: weak aug; images[1]: strong aug
        # targets: labels for images
        if i_batch < 3:
            plt.figure()
            plt.imshow(restore_stats(utils.make_grid(images[0]), cfg, TRANSFORM_CIFAR))
            plt.imshow(restore_stats(utils.make_grid(images[1]), cfg, TRANSFORM_CIFAR))
            print([classes[i] for i in targets.numpy()])
            plt.axis('off')
            plt.ioff()
            plt.show()
        else:
            break

    # DATA LOADER TEST FOR LABELED DATA
    for i_batch, sample_batched in enumerate(lb_tfm_dataloader):
        images, targets = sample_batched
        print(i_batch, images.size(), targets.size())
        if i_batch < 3:
            plt.figure()
            plt.imshow(restore_stats(utils.make_grid(images), cfg,TRANSFORM_CIFAR))
            print([classes[i] for i in targets.numpy()])
            plt.axis('off')
            plt.ioff()
            plt.show()
        else:
            break

    # DATA LOADER TEST FOR TEST DATA
    for i_batch, sample_batched in enumerate(tst_dataloader):
        images, targets = sample_batched
        print(i_batch, images.size(), targets.size())
        if i_batch < 3:
            plt.figure()
            plt.imshow(restore_stats(utils.make_grid(images), cfg,TRANSFORM_CIFAR))
            print([classes[i] for i in targets.numpy()])
            plt.axis('off')
            plt.ioff()
            plt.show()
        else:
            break