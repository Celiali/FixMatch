import math
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from os.path import exists
from os import mkdir
import shutil
import logging

import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import ignite

from utils.utils import accuracy,AverageMeter
from utils.ema import EMA


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    """ <Borrowed from `transformers`>
        Create a schedule with a learning rate that decreases from the initial lr set in the optimizer to 0,
        after a warmup period during which it increases from 0 to the initial lr set in the optimizer.
        Args:
            optimizer (:class:`~torch.optim.Optimizer`): The optimizer for which to schedule the learning rate.
            num_warmup_steps (:obj:`int`): The number of steps for the warmup phase.
            num_training_steps (:obj:`int`): The total number of training steps.
            last_epoch (:obj:`int`, `optional`, defaults to -1): The index of the last epoch when resuming training.
        Return:
            :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))  # this is correct
    return LambdaLR(optimizer, _lr_lambda, last_epoch)

logger = logging.getLogger(__name__)

class VanillaExperiment(object):
    def __init__(self, wideresnet, params):
        self.model = wideresnet
        self.params = params

        # optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.params.optim_lr,
                          momentum=self.params.optim_momentum, nesterov=self.params.used_nesterov)

        per_epoch_steps = self.params.n_imgs_per_epoch // self.params.batch_size
        total_training_steps = self.params.epoch_n * per_epoch_steps
        warmup_steps = self.params.warmup * per_epoch_steps
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, total_training_steps)

        #save log
        self.summary_logdir = os.path.join(self.params.log_path, 'summaries')

        # used Gpu or not
        self.used_gpu = self.params.used_gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.used_gpu else 'cpu')

        # used EWA or not
        self.ema = self.params.ema_used
        if self.ema:
            self.ema_model = EMA(self.model, self.params.ema_decay)
            logger.info("[EMA] initial ")

    def forward(self, input):
        return self.model(input)

    def train_step(self):
        logger.info("***** Running training *****")
        start = time.time()
        batch_time_meter = AverageMeter()
        train_losses_meter = AverageMeter()

        # turn on model training
        self.model.train()
        for batch_idx, (inputs_labelled, targets_labelled) in enumerate(self.labelled_loader):
            if self.used_gpu:
                inputs_labelled = inputs_labelled.to(device = self.device)
                targets_labelled = targets_labelled.to(device=self.device)

            # forward
            outputs_labelled = self.forward(inputs_labelled)

            # compute loss for labelled data,unlabeled data,total loss
            loss = F.cross_entropy(outputs_labelled, targets_labelled, reduction='mean')

            # compute gradient and backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # updating ema model (params)
            if self.ema:
                self.ema_model.update_params()
                # logger.info("[EMA] update params()")

            ############ update recording #################
            train_losses_meter.update(loss.item())
            batch_time_meter.update(time.time() - start)

        # updating ema model (buffer)
        if self.ema:
            self.ema_model.update_buffer()
            logger.info("[EMA] update buffer()")

        return train_losses_meter.avg

    def validation_step(self):
        logger.info("***** Running validation *****")
        start = time.time()
        batch_time_meter = AverageMeter()
        valid_losses_meter = AverageMeter()
        top1_meter = AverageMeter()
        top5_meter = AverageMeter()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.valid_dataloader):
                self.model.eval()
                if self.used_gpu:
                    inputs = inputs.to(device=self.device)
                    targets = targets.to(device=self.device)
                # forward
                outputs = self.forward(inputs)
                # compute loss and accuracy
                loss = F.cross_entropy(outputs, targets)
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                # update recording
                valid_losses_meter.update(loss.item(), inputs.shape[0])
                top1_meter.update(acc1[0], inputs.size(0))
                top5_meter.update(acc5[0], inputs.size(0))
                batch_time_meter.update(time.time() - start)
        return valid_losses_meter.avg,top1_meter.avg,top5_meter.avg

    def test_step(self):
        logger.info("***** Running testing *****")
        start = time.time()
        batch_time_meter = AverageMeter()
        test_losses_meter = AverageMeter()
        top1_meter = AverageMeter()
        top5_meter = AverageMeter()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_dataloader):
                self.model.eval()
                if self.used_gpu:
                    inputs = inputs.to(device=self.device)
                    targets = targets.to(device=self.device)
                # forward
                outputs = self.forward(inputs)
                # compute loss and accuracy
                loss = F.cross_entropy(outputs, targets)
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                # update recording
                test_losses_meter.update(loss.item(), inputs.shape[0])
                top1_meter.update(acc1[0], inputs.size(0))
                top5_meter.update(acc5[0], inputs.size(0))
                batch_time_meter.update(time.time() - start)
        return test_losses_meter.avg,top1_meter.avg,top5_meter.avg


    def testing(self):
        # apply shadow model
        if self.ema:
            self.ema_model.apply_shadow()
            logger.info("[Testing with EMA] apply shadow")

        test_loss, top1_acc, top5_acc= self.test_step()
        if self.ema:
            logger.info(
                "[Testing(EMA)] testing_loss: {:.4f}, test Top1 acc:{}, test Top5 acc: {}".format(test_loss,top1_acc,top5_acc))
        else:
            logger.info(
            "[Testing] testing_loss: {:.4f}, test Top1 acc:{}, test Top5 acc: {}".format(test_loss,top1_acc,top5_acc))

        # restore the params
        if self.ema:
            self.ema_model.restore()
            logger.info("[EMA] restore ")


    def fitting(self):
        self.swriter = SummaryWriter(log_dir=self.summary_logdir)
        prev_lr = np.inf

        if self.params.resume:
            # optionally resume from a checkpoint
            start_epoch = self.resume_model()
        else:
            start_epoch = 0

        for epoch_idx in range(start_epoch, self.params.epoch_n):
            # turn on training
            start = time.time()
            train_loss = self.train_step()
            end = time.time()

            cur_lr = self.optimizer.param_groups[0]['lr']
            if cur_lr != prev_lr:
                prev_lr = cur_lr

            # apply shadow model
            if self.ema:
                self.ema_model.apply_shadow()
                logger.info("[EMA] apply shadow")

            # validation
            valid_loss, top1_acc, top5_acc= self.validation_step()

            # restore the params
            if self.ema:
                self.ema_model.restore()
                logger.info("[EMA] restore ")

            # saving data in tensorboard
            self.swriter.add_scalars('train/loss', {'train_loss': train_loss, 'test_loss': valid_loss}, epoch_idx)
            self.swriter.add_scalars('test/accuracy', {'Top1': top1_acc, 'Top5': top5_acc}, epoch_idx)
            self.swriter.add_scalars('train/lr', {'Current Lr': cur_lr}, epoch_idx)

            logger.info(
                "Epoch {}. time:{} seconds, lr:{:.4f}, "
                "train_loss: {:.4f}, testing_loss: {:.4f}, test Top1 acc:{}, test Top5 acc: {}".format(
                    epoch_idx, end - start, cur_lr, train_loss, valid_loss, top1_acc, top5_acc))

            # saving model
            if epoch_idx == 0 or (epoch_idx + 1) % self.params.save_every == 0:
                state = {
                    'epoch': epoch_idx,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'ema_state_dict': self.ema_model.shadow if self.ema else None
                }
                self.save_checkpoint(state, epoch_idx)

        # close tensorboard
        self.swriter.close()

    def labelled_loader(self, labelled_training_dataset):
        self.num_train = len(labelled_training_dataset)
        self.labelled_loader = DataLoader(labelled_training_dataset,
                                           batch_size=self.params.batch_size,
                                           shuffle=True,
                                           drop_last=True)
        logger.info("Loading Labelled Loader")
        return

    def unlabelled_loader(self,unlabelled_training_dataset, mu):
        self.num_valid = len(unlabelled_training_dataset)
        self.unlabelled_loader = DataLoader(unlabelled_training_dataset,
                                           batch_size=self.params.batch_size * mu,
                                           shuffle=True,
                                           drop_last=True)
        logger.info("Loading Unlabelled Loader")
        return

    def validation_loader(self,valid_dataset):
        self.num_valid_imgs = len(valid_dataset) # same as len(dataloader)
        self.valid_dataloader = DataLoader(valid_dataset,
                                          batch_size=self.params.batch_size,
                                          shuffle=False,
                                          drop_last=True)
        logger.info("Loading Validation Loader")
        return

    def test_loader(self,test_dataset):
        self.num_test_imgs = len(test_dataset) # same as len(dataloader)
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=self.params.batch_size,
                                          shuffle=False,
                                          drop_last=True)
        logger.info("Loading testing Loader")
        return

    def load_model(self, mdl_fname, cuda=False):
        if self.used_gpu:
            self.model.load_state_dict(torch.load(mdl_fname))
            self.model.cuda()
        else:
            self.model.load_state_dict(torch.load(mdl_fname, map_location='cpu'))
        self.model.eval()

    def resume_model(self):
        """ optionally resume from a checkpoint
        Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262 """
        start_epoch = 0
        best_acc = 0.0
        if self.params.resume:
            if os.path.isfile(self.params.resume_checkpoints):
                print("=> loading checkpoint '{}'".format(self.params.resume_checkpoints))
                logger.info("==> Resuming from checkpoint..")
                if self.used_gpu:
                    # Map model to be loaded to specified single gpu.
                    checkpoint = torch.load(self.params.resume_checkpoints, map_location=self.device)
                else:
                    checkpoint = torch.load(self.params.resume_checkpoints)
                start_epoch = checkpoint['epoch']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                if self.ema:
                    self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})".format(self.params.resume_checkpoints, checkpoint['epoch']))
                logger.info(
                    "=> loaded checkpoint '{}' (epoch {})".format(self.params.resume_checkpoints, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.params.resume_checkpoints))
                logger.info("=> no checkpoint found at '{}'".format(self.params.resume_checkpoints))
        return start_epoch

    def save_checkpoint(self,state, epoch_idx):
        saving_checkpoint_file_folder = os.path.join(self.params.out_model,self.params.log_path.split('/')[-1])
        if not exists(saving_checkpoint_file_folder):
            mkdir(saving_checkpoint_file_folder)
        filename = os.path.join(saving_checkpoint_file_folder,'{}_epoch_{}.pth.tar'.format(self.params.name, epoch_idx))
        ######## testing #########
        # torch.save(state, filename)
        logger.info("[Checkpoints] Epoch {}, saving to {}".format(state['epoch'], filename))
