import yaml
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from easydict import EasyDict as edict
import random

from datasets import *
from models import *
from experiments import *
from utils.utils import setup_default_logging

if __name__ == '__main__':
    #### Error: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.###
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # configuration
    parser = argparse.ArgumentParser(description='Generic runner for FixMatch')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='config/config.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config_file = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    CONFIG = edict(config_file)
    print('==> CONFIG is: \n', CONFIG, '\n')

    # initial logging file
    logger = setup_default_logging(CONFIG, string = 'Train')
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
    labeled_training_dataset,unlabeled_training_dataset, test_dataset = data.get_dataset()

    # build wideresnet
    model = WRN_MODELS[CONFIG.MODEL.name](CONFIG.MODEL)
    logger.info("[Model] Building model {}".format(CONFIG.MODEL.name))

    if CONFIG.EXPERIMENT.used_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device=device)

    experiment = EXPERIMENT[CONFIG.EXPERIMENT.name](model,CONFIG.EXPERIMENT)
    experiment.labelled_loader(labeled_training_dataset)
    if CONFIG.DATASET.loading_data != 'LOAD_ORIGINAL' and unlabeled_training_dataset != None:
        experiment.unlabelled_loader(unlabeled_training_dataset, CONFIG.DATASET.mu)
    experiment.valid_loader(test_dataset)
    experiment.fitting()
    print("======= Training done =======")
    logger.info("======= Training done =======")
    experiment.test_loader(test_dataset)
    experiment.testing()
    print("======= Testing done =======")
    logger.info("======= Testing done =======")