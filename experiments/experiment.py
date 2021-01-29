import math
import os
import time
import numpy as np
from os.path import exists
from os import mkdir
import logging


import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from utils.utils import accuracy,AverageMeter, save_cfmatrix, nl_loss
from utils.ema import EMA
from datasets.datasets1 import BatchWeightedRandomSampler
from augmentations.ctaugment import deserialize

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

class NegEntropy(object):
    ### Import from https://github.com/LiJunnan1992/DivideMix/blob/d9d3058fa69a952463b896f84730378cdee6ec39/Train_cifar.py#L205
    def __init__(self, equal_freq=False):
        
        if equal_freq:
            self.loss_func = lambda x: torch.sum((torch.mean(x, dim=0)+1e-5).log()* x)
        else:
            self.loss_func = lambda x: torch.mean(torch.sum((x+1e-5).log()*x, dim=1)) 


    def __call__(self,outputs, ):
        probs = torch.softmax(outputs, dim=1)
        return self.loss_func(probs)

logger = logging.getLogger(__name__)

class FMExperiment(object):
    def __init__(self, wideresnet, params, cta=None):
        self.model = wideresnet
        self.params = params
        self.cta = cta
        self.save_cfmatrix = params.save_cfmatrix
        self.curr_device = None
        # optimizer
        # refer to https://github.com/kekmodel/FixMatch-pytorch/blob/248268b8e6777de4f5c8768ee7fc53c4f4c8a13c/train.py#L237
        no_decay = ['bias', 'bn']
        grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.params.wdecay},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = optim.SGD(grouped_parameters, lr=self.params.optim_lr,
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

        if self.params.neg_penalty:
            self.conf_penalty = NegEntropy(self.params.equal_freq)

    def forward(self, input):
        return self.model(input)

    def update_cta(self, data): 
        (cta_imgs, policies), cat_targets = data # labeled
        if self.ema:
            model = self.ema_model
        else:
            model = self.model
        self.model.eval() # 
        with torch.no_grad():
            cta_imgs = cta_imgs.to(self.device)
            logits = self.forward(cta_imgs)
            probs = torch.softmax(logits, dim=1)
            policies = [deserialize(p) for p in policies]
            for prob, t, policy in zip(probs, cat_targets, policies):
                prob[t] -= 1
                prob = torch.abs(prob).sum()
                self.cta.update_rates(policy, 1.0 - 0.5 * prob.item())
        self.model.train()



    def train_step(self):
        logger.info("***** Running training *****")
        start = time.time()
        batch_time_meter = AverageMeter()
        train_losses_meter = AverageMeter()
        labelled_losses_meter = AverageMeter()
        unlabeled_losses_meter = AverageMeter() # pseudo loss for unlabeled data with strong augmentation
        mask_meter = AverageMeter()

        # path for saving confusion matrix
        now = int(round(time.time()*1000))
        now = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(now/1000))
        save_to = self.params.log_path + '/%s_'%now

        #### optional value cal
        unlabeled_losses_real_strong_meter = AverageMeter()# real loss for unlabeled data with strong augmentation
        corrrect_unlabeled_num_meter = AverageMeter()#Num of Correct Predicted for Unlabelled data
        pro_above_threshold_num_meter = AverageMeter()#Num of gradient-included Unlabelled data
        unlabelled_weak_top1_acc_meter = AverageMeter()
        unlabelled_weak_top5_acc_meter = AverageMeter()

        # turn on model training
        train_loader = zip(self.labelled_loader, self.unlabelled_loader)
        if self.cta:
            cta_iter = iter(self.cta_probe_dataloader)
        self.model.train()
        for batch_idx, (data_labelled, data_unlabelled) in enumerate(train_loader):
            inputs_labelled, targets_labelled = data_labelled
            (inputs_unlabelled_weak, inputs_unlabelled_strong), targets_unlabelled = data_unlabelled
            # print('targets_labeled\n', [targets_labelled[np.where(targets_labelled==i)].shape for i in range(10)])
            # print('targets_unlabelled\n', [targets_unlabelled[np.where(targets_unlabelled==i)].shape for i in range(10)])
            if self.used_gpu:
                inputs_labelled = inputs_labelled.to(device = self.device)
                targets_labelled = targets_labelled.to(device=self.device)
                inputs_unlabelled_weak = inputs_unlabelled_weak.to(device = self.device)
                inputs_unlabelled_strong = inputs_unlabelled_strong.to(device = self.device)
                targets_unlabelled = targets_unlabelled.to(device=self.device)

            batch_size = inputs_labelled.shape[0]
            inputs = torch.cat((inputs_labelled, inputs_unlabelled_weak, inputs_unlabelled_strong))

            # forward
            outputs = self.forward(inputs)
            # separate different outputs for different inputs
            outputs_labelled = outputs[:batch_size]
            outputs_unlabelled_weak, outputs_unlabelled_strong = outputs[batch_size:].chunk(2)
            del outputs

            # compute pseudo label for unlabeled data with weak augmentations
            outputs_labelled_weak_pro = torch.softmax(outputs_unlabelled_weak.detach(), dim=-1)
            scores, pseudo_label = torch.max(outputs_labelled_weak_pro, dim=-1)
            mask = scores.ge(self.params.threshold).float()

            # compute loss for labelled data,unlabeled data,total loss
            loss_labelled = F.cross_entropy(outputs_labelled, targets_labelled, reduction='mean')
            
            if self.params.use_nlloss:
                n_class = targets_labelled.max()+1
                loss_unlabelled = nl_loss(outputs_unlabelled_strong, F.one_hot(pseudo_label, num_classes=n_class), self.params.q)
                # loss_unlabelled_ce = F.cross_entropy(outputs_unlabelled_strong, pseudo_label,reduction='none')  
            else: 
                loss_unlabelled = F.cross_entropy(outputs_unlabelled_strong, pseudo_label,reduction='none')  

            loss_unlabelled_guess = (loss_unlabelled * mask).mean()
            loss = loss_labelled + self.params.lambda_unlabeled * loss_unlabelled_guess
            if self.params.neg_penalty:
                penalty = self.conf_penalty(outputs_labelled) ## for labelled data
                if self.params.eta_dynamic:
                    t = math.cos((math.pi * 7. * self.optimizer._step_count) / (16.*1024 * self.params.epoch_n))
                else:
                    t = 1
                loss = loss + self.params.eta_negpenalty * t * penalty

            # compute gradient and backprop
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip)
            self.optimizer.step()
            self.scheduler.step()

            # update cta bin weights
            if self.cta:
                self.update_cta(next(cta_iter))

            # updating ema model (params)
            if self.ema:
                self.ema_model.update_params()
                # logger.info("[EMA] update params()")

            #### optional value cal
            # true loss for unlabelled data with strong augmentation --- without any threshold
            loss_unlabelled_true_strong = F.cross_entropy(outputs_unlabelled_strong, targets_unlabelled)
            # correct predicted num --- with threshold
            corrrect_unlabeled_num = ((pseudo_label == targets_unlabelled).float() * mask).sum()
            # numbers of predicted pro above threshold
            pro_above_threshold_num = mask.sum()
            # pseudo label accuracy are cal without threshold
            weak_top1_acc,weak_top5_acc = accuracy(outputs_unlabelled_weak,targets_unlabelled, topk=(1,5))

            ############ update recording #################
            train_losses_meter.update(loss.item())
            labelled_losses_meter.update(loss_labelled.item())
            unlabeled_losses_meter.update(loss_unlabelled_guess.item())
            mask_meter.update(mask.mean().item())
            #### optional value cal
            unlabeled_losses_real_strong_meter.update(loss_unlabelled_true_strong.item())
            corrrect_unlabeled_num_meter.update(corrrect_unlabeled_num.item())
            pro_above_threshold_num_meter.update(pro_above_threshold_num.item())
            unlabelled_weak_top1_acc_meter.update(weak_top1_acc.item())
            unlabelled_weak_top5_acc_meter.update(weak_top5_acc.item())
            batch_time_meter.update(time.time() - start)

            # save confusion matrix every 100 steps
            if self.save_cfmatrix and batch_idx % self.params.save_matrix_every == 0: #                 
                outputs_labelled = torch.argmax(outputs_labelled, dim=-1)
                save_cfmatrix(outputs_labelled, pseudo_label, mask, targets_labelled, targets_unlabelled, save_to=save_to, comment='step%d'%batch_idx)

        # updating ema model (buffer)
        if self.ema:
            self.ema_model.update_buffer()
            logger.info("[EMA] update buffer()")

        return train_losses_meter.avg,labelled_losses_meter.avg,unlabeled_losses_meter.avg,mask_meter.avg,unlabeled_losses_real_strong_meter.avg,\
               corrrect_unlabeled_num_meter.sum,pro_above_threshold_num_meter.sum,unlabelled_weak_top1_acc_meter.avg,unlabelled_weak_top5_acc_meter.avg

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
                top1_meter.update(acc1.item(), inputs.shape[0])
                top5_meter.update(acc5.item(), inputs.shape[0])
                batch_time_meter.update(time.time() - start)

        return test_losses_meter.avg,top1_meter.avg,top5_meter.avg

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
                top1_meter.update(acc1.item(), inputs.shape[0])
                top5_meter.update(acc5.item(), inputs.shape[0])
                batch_time_meter.update(time.time() - start)
        return valid_losses_meter.avg,top1_meter.avg,top5_meter.avg


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
        # initial tensorboard
        self.swriter = SummaryWriter(log_dir=self.summary_logdir)

        if self.params.resume:
            # optionally resume from a checkpoint
            start_epoch = self.resume_model()
        else:
            start_epoch = 0

        prev_lr = np.inf
        for epoch_idx in range(start_epoch, self.params.epoch_n):
            if self.params.save_cfmatrix and (epoch_idx % self.params.save_matrix_every == 0 or epoch_idx == self.params.epoch_n-1):
                self.save_cfmatrix = True
            else: self.save_cfmatrix = False

            # turn on training
            start = time.time()
            train_loss,labelled_loss,unlabeled_loss,mask,unlabeled_losses_real_strong,\
               corrrect_unlabeled_num,pro_above_threshold_num,unlabelled_weak_top1_acc,unlabelled_weak_top5_acc = self.train_step()
            end = time.time()
            print("epoch {}: use {} seconds".format(epoch_idx, end - start))

            cur_lr = self.optimizer.param_groups[0]['lr'] # self.scheduler.get_last_lr()[0]
            if cur_lr != prev_lr:
                print('--- Optimizer learning rate changed from %.2e to %.2e ---' % (prev_lr, cur_lr))
                prev_lr = cur_lr

            # testing
            raw_test_loss, raw_top1_acc, raw_top5_acc = self.validation_step()

            # apply shadow model
            if self.ema:
                self.ema_model.apply_shadow()
                logger.info("[EMA] apply shadow")

                # testing
                ema_test_loss, ema_top1_acc, ema_top5_acc= self.validation_step()

                # restore the params
                self.ema_model.restore()
                logger.info("[EMA] restore ")

            # saving data in tensorboard
            self.swriter.add_scalars('train/lr', {'Current Lr': cur_lr}, epoch_idx)
            self.swriter.add_scalars('test/accuracy', {'Top1': raw_top1_acc, 'Top5': raw_top5_acc}, epoch_idx)
            if self.ema:
                self.swriter.add_scalars('train/loss', {'train_loss': train_loss, 'raw_test_loss': raw_test_loss,}, epoch_idx)
                self.swriter.add_scalars('train_w_ema/loss', {'train_loss': train_loss, 'ema_test_loss': ema_test_loss}, epoch_idx)
                self.swriter.add_scalars('test_w_ema/accuracy', {'Top1': ema_top1_acc, 'Top5': ema_top5_acc}, epoch_idx)
                logger.info("[EMA] Epoch {}.[Train] time:{} seconds, lr:{:.4f}, train_loss: {:.4f}, "
                            "unlabeled_losses_real_strong:{:.4f},corrrect_unlabeled_num:{},pro_above_threshold_num:{},"
                            "unlabelled_weak_top1_acc:{},unlabelled_weak_top5_acc:{}  ".format(epoch_idx, end - start,
                                                                                               cur_lr, train_loss,
                                                                                               unlabeled_losses_real_strong,
                                                                                               corrrect_unlabeled_num,
                                                                                               pro_above_threshold_num,
                                                                                               unlabelled_weak_top1_acc,
                                                                                               unlabelled_weak_top5_acc))
                logger.info(
                    "[EMA] Epoch {}. [Validation] raw_testing_loss: {:.4f}, raw test Top1 acc:{}, raw test Top5 acc: {}, ema_testing_loss: {:.4f}, ema test Top1 acc:{}, ema test Top5 acc: {}".format(
                        epoch_idx, raw_test_loss, raw_top1_acc, raw_top5_acc, ema_test_loss, ema_top1_acc, ema_top5_acc))

            else:   # if ema is not used ; save the infor without ema
                self.swriter.add_scalars('train/loss', {'train_loss': train_loss, 'test_loss': raw_test_loss,}, epoch_idx)
                logger.info(
                    "[no EMA] Epoch {}. [Train] time:{} seconds, lr:{:.4f}, "
                    "train_loss: {:.4f}, unlabeled_losses_real_strong:{:.4f},"
                    "corrrect_unlabeled_num:{},pro_above_threshold_num:{},"
                    "unlabelled_weak_top1_acc:{},unlabelled_weak_top5_acc:{}  ".format(epoch_idx, end - start, cur_lr, train_loss,
                                 unlabeled_losses_real_strong, corrrect_unlabeled_num,pro_above_threshold_num,unlabelled_weak_top1_acc,unlabelled_weak_top5_acc))
                logger.info("[no EMA] Epoch {}. [Validation] testing_loss: {:.4f}, test Top1 acc:{}, test Top5 acc: {}".format(epoch_idx,
                        raw_test_loss, raw_top1_acc, raw_top5_acc))


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
        if self.params.batch_balanced:
            kwargs = dict(
                batch_sampler= BatchWeightedRandomSampler(labelled_training_dataset, batch_size=self.params.batch_size) if self.params.batch_balanced else None,
            )
        else: 
            kwargs =dict(
                batch_size=self.params.batch_size,
                shuffle=True,
                drop_last=True
            )
        self.labelled_loader = DataLoader(labelled_training_dataset, num_workers=self.params.num_workers, pin_memory=True, **kwargs)
        logger.info("Loading Labelled Loader")
        return

    def unlabelled_loader(self,unlabelled_training_dataset, mu):
        self.num_valid = len(unlabelled_training_dataset)
        if self.params.batch_balanced:
            kwargs = dict(
                batch_sampler= BatchWeightedRandomSampler(unlabelled_training_dataset, batch_size=self.params.batch_size * mu) if self.params.batch_balanced else None,
            )
        else: 
            kwargs =dict(
                batch_size=self.params.batch_size * mu,
                shuffle=True,
                drop_last=True,
            )
        self.unlabelled_loader = DataLoader(unlabelled_training_dataset, num_workers=self.params.num_workers, pin_memory=True, **kwargs)
        logger.info("Loading Unlabelled Loader")
        return

    def validation_loader(self,valid_dataset):
        self.num_valid_imgs = len(valid_dataset) # same as len(dataloader)
        self.valid_dataloader = DataLoader(valid_dataset,
                                          batch_size=self.params.batch_size,
                                          shuffle=False,
                                          drop_last=False,
                                          num_workers=self.params.num_workers,
                                          pin_memory=True)
        logger.info("Loading Validation Loader")
        return

    def test_loader(self,test_dataset):
        self.num_test_imgs = len(test_dataset) # same as len(dataloader)
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=self.params.batch_size,
                                          shuffle=False,
                                          drop_last=False, 
                                          num_workers=self.params.num_workers, 
                                          pin_memory=True)
        logger.info("Loading Testing Loader")
        return

    def cta_probe_loader(self, cta_probe_dataset):
        self.cta_probe_dataloader = DataLoader(cta_probe_dataset,
                                          batch_size=self.params.batch_size,
                                          shuffle=False,
                                          drop_last=False, 
                                          num_workers=self.params.num_workers, 
                                          pin_memory=True)
        logger.info("Loading cta probe Loader")
        return 

    def load_model(self, mdl_fname, cuda=False):
        if self.used_gpu:
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(mdl_fname, map_location=self.device)
        else:
            checkpoint = torch.load(mdl_fname)
        self.model.load_state_dict(checkpoint['state_dict'])
        if self.ema:
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
        self.model.eval()
        logger.info("Loading previous model")

    def resume_model(self):
        # TODO: not test
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
                if checkpoint['ema_state_dict']:
                    self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
                elif self.ema:
                    self.ema_model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})".format(self.params.resume_checkpoints, checkpoint['epoch']))
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(self.params.resume_checkpoints, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.params.resume_checkpoints))
                logger.info("=> no checkpoint found at '{}'".format(self.params.resume_checkpoints))
        return start_epoch

    def save_checkpoint(self,state, epoch_idx):
        saving_checkpoint_file_folder = os.path.join(self.params.out_model, self.params.log_path.split('/')[-1])
        if not exists(saving_checkpoint_file_folder):
            mkdir(saving_checkpoint_file_folder)
        filename = os.path.join(saving_checkpoint_file_folder,'{}_epoch_{}.pth.tar'.format(self.params.name, epoch_idx))
        torch.save(state, filename)
        logger.info("[Checkpoints] Epoch {}, saving to {}".format(state['epoch'], filename))

# def end_writer(self):
    #     self.swriter.close()
