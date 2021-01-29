import yaml
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from easydict import EasyDict as edict
import random
import glob
from omegaconf import DictConfig, OmegaConf
import hydra

from datasets import *
from models import *
from experiments import *
from utils.utils import setup_default_logging


    #### Error: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.###
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # configuration


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

    # checkpointslist = sorted(glob.glob('/Users/cil/Documents/DL_advanced/results/barely10/checkpoints/barely10/*.pth.tar'))
    checkpointslist = sorted(
        glob.glob(CONFIG.EXPERIMENT.resume_checkpoints + '*.pth.tar'))
    # checkpointslist = sorted(glob.glob('/Users/cil/Documents/DL_advanced/results/reproduce/checkpoints/exp40_origin/*.pth.tar'))
    # checkpointslist = sorted(glob.glob('/Users/cil/Documents/DL_advanced/results/exp_noise/checkpoints/exp_1loss_noise/*.pth.tar'))
    for i in checkpointslist:
        # CONFIG.DATASET.label_num = int(i.split('/')[-2].split("_")[0][3:])
        CONFIG.EXPERIMENT.resume_checkpoints = i
        # logger.info("[Experiment {}]".format(CONFIG.DATASET.label_num))
        if int(i.split('/')[-1].split('.')[0].split('_')[-1]) < 60:
            continue
        logger.info("[Resume Checkpoints] {}".format(i))
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
        all_dataset = data.get_dataset()
        test_dataset = all_dataset[3]

        # build wideresnet
        model = WRN_MODELS[CONFIG.MODEL.name](CONFIG.MODEL)
        logger.info("[Model] Building model {}".format(CONFIG.MODEL.name))

        if CONFIG.EXPERIMENT.used_gpu:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device=device)

        experiment = EXPERIMENT[CONFIG.EXPERIMENT.name](
            model, CONFIG.EXPERIMENT)
        experiment.load_model(CONFIG.EXPERIMENT.resume_checkpoints)
        experiment.test_loader(test_dataset)
        experiment.testing()
        print("======= Testing done =======")
        logger.info("======= Testing done =======")

if __name__ == '__main__':
    main()