# import yaml
# import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from easydict import EasyDict as edict
from sklearn.model_selection import KFold
from omegaconf import DictConfig, OmegaConf
import hydra
from torch.utils.data import ConcatDataset

from datasets import *
from models import *
from experiments import *
from utils.utils import setup_default_logging

K_OUTER_FOLD = 3
K_INNER_FOLD = 10


@hydra.main(config_path='./config', config_name='config')
def main(CONFIG: DictConfig) -> None:
    # # configuration
    # parser = argparse.ArgumentParser(description='Generic runner for FixMatch')
    # parser.add_argument('--config',  '-c',
    #                     dest="filename",
    #                     metavar='FILE',
    #                     help =  'path to the config file',
    #                     default='config/config.yaml')

    # args = parser.parse_args()
    # with open(args.filename, 'r') as file:
    #     try:
    #         config_file = yaml.safe_load(file)
    #     except yaml.YAMLError as exc:
    #         print(exc)
    # CONFIG = edict(config_file)
    print('==> CONFIG is: \n', OmegaConf.to_yaml(CONFIG), '\n')

    # initial logging file
    logger = setup_default_logging(CONFIG, string='Train')
    logger.info(CONFIG)

    # # For reproducibility, set random seed
    if CONFIG.Logging.seed == 'None':
        CONFIG.Logging.seed = random.randint(1, 10000)
    random.seed(CONFIG.Logging.seed)
    np.random.seed(CONFIG.Logging.seed)
    torch.manual_seed(CONFIG.Logging.seed)
    torch.cuda.manual_seed_all(CONFIG.Logging.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # get datasets
    data = LOADDATA[CONFIG.DATASET.loading_data](CONFIG.DATASET)

    cta = data.get_cta() if CONFIG.DATASET.strongaugment == 'CTA' else None

    logger.info("[Model] Building model {}".format(CONFIG.MODEL.name))

    trainset, testset = data.get_vanila_dataset()

    num_train = len(trainset)
    num_each_outer_fold = num_train // K_OUTER_FOLD
    leftover_outer = num_train - num_each_outer_fold * K_OUTER_FOLD

    outer_folds = torch.utils.data.random_split(trainset, [num_each_outer_fold] * (K_OUTER_FOLD - 1) +
                                                [num_each_outer_fold + leftover_outer])

    for curr_outer_fold_idx in range(K_OUTER_FOLD):
        print(f'Entering outer fold {curr_outer_fold_idx}')
        # leave one out
        outer_fold_train_dataset = ConcatDataset([dtst for idx, dtst in enumerate(outer_folds) if idx != curr_outer_fold_idx])
        outer_fold_val_dataset = outer_folds[curr_outer_fold_idx]

        num_inner = len(outer_fold_train_dataset)
        num_each_inner_fold = num_inner // K_INNER_FOLD
        leftover_inner = num_inner - num_each_inner_fold * K_INNER_FOLD

        inner_folds = torch.utils.data.random_split(outer_fold_train_dataset, [num_each_inner_fold] * (K_INNER_FOLD - 1)
                                                    + [num_each_inner_fold + leftover_inner])
        best_model, best_model_top1_acc = None, None
        for curr_inner_fold_idx in range(K_INNER_FOLD):
            print(f'Entering inner fold {curr_inner_fold_idx}')

            # build the simple CNN
            # model = WRN_MODELS['SimpleColorCNN'](CONFIG.MODEL)

            # build wideresnet
            model = WRN_MODELS[CONFIG.MODEL.name](CONFIG.MODEL)

            if CONFIG.EXPERIMENT.used_gpu:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = model.to(device=device)

            experiment = EXPERIMENT[CONFIG.EXPERIMENT.name](
                model, CONFIG.EXPERIMENT, cta)

            inner_fold_train_dataset = ConcatDataset([dtst for idx, dtst in enumerate(inner_folds) if idx != curr_inner_fold_idx])
            inner_fold_val_dataset = inner_folds[curr_inner_fold_idx]

            if cta:
                labeled_training_dataset, unlabeled_training_dataset, valid_dataset, cta_dataset = data.vanilla_dataset_to_unlabeled(inner_fold_train_dataset)
            else:
                labeled_training_dataset, unlabeled_training_dataset, valid_dataset = data.vanilla_dataset_to_unlabeled(inner_fold_train_dataset)

            experiment.labelled_loader(labeled_training_dataset)
            if CONFIG.DATASET.loading_data != 'LOAD_ORIGINAL' and unlabeled_training_dataset != None:
                experiment.unlabelled_loader(
                    unlabeled_training_dataset, CONFIG.DATASET.mu)
            experiment.validation_loader(valid_dataset)
            experiment.fitting()
            logger.info(f"======= Training done outer fold {curr_outer_fold_idx} inner fold {curr_inner_fold_idx} =======")
            experiment.test_loader(inner_fold_val_dataset)
            test_loss, top1_acc, top5_acc = experiment.testing()

            if best_model is None or top1_acc > best_model_top1_acc:
                best_model = model
                best_model_top1_acc = top1_acc

            print(f'top1: {top1_acc}, top5: {top5_acc}')
            logger.info("======= Validation done =======")

        experiment = EXPERIMENT[CONFIG.EXPERIMENT.name](
            best_model, CONFIG.EXPERIMENT, cta)
        experiment.test_loader(outer_fold_val_dataset)
        test_loss, top1_acc, top5_acc = experiment.testing()
        print(f'OUTER FOLD {curr_outer_fold_idx} TEST')
        print(f'top1: {top1_acc}, top5: {top5_acc}')


if __name__ == '__main__':
    # ### Error: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.###
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()
