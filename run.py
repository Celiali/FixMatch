import yaml
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from easydict import EasyDict as edict

from datasets.datasets import *
from models.wideresnet import *
from experiments.experiment import *

if __name__ == '__main__':
    #### Error: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.###
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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

    # # For reproducibility
    torch.manual_seed(CONFIG.Logging.seed)
    np.random.seed(CONFIG.Logging.seed)
    # cudnn.deterministic = True
    # cudnn.benchmark = False

    # get datasets
    data = LoadDataset(CONFIG.DATASET)
    labeled_training_dataset, test_dataset = data.get_dataset()

    # build wideresnet
    model = build_wideresnet(CONFIG.MODEL)

    if CONFIG.EXPERIMENT.used_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device=device)

    experiment = FMExperiment(model,CONFIG.EXPERIMENT)
    experiment.labelled_loader(labeled_training_dataset)
    # experiment.unlabelled_loader(unlabeled_training_dataset)
    experiment.test_loader(test_dataset)
    experiment.fitting()

    experiment.end_writer()
    print("======= Training done =======")