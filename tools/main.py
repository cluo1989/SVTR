'''
Author: Cristiano-3 chunanluo@126.com
Date: 2023-03-20 14:24:05
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2023-04-20 18:46:48
FilePath: /SVTR/tools/train.py
Description: 
'''
# coding: utf-8
import time
import yaml
import argparse
from easydict import EasyDict as edict
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from datasets.rec_dataset import RecDataset
from utils import AverageMeter, ProgressMeter, Summary
from modeling.metrics.rec_metric import accuracy


def train(train_loader, model, criterion, optimizer, epoch, device, config):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accs = AverageMeter('Acc', ':6.2f')
    
    progress = ProgressMeter(
        len(train_loader), 
        [batch_time, data_time, losses], 
        prefix='Epoch: [{}]'.format(epoch)
        )
    
    # train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure acc and record loss
        acc = accuracy(output, target)
        losses.update(loss.item(), images.size(0))
        accs.update(acc, images.size(0))

        # compute gradient and SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % config.PRINT_FREQ == 0:
            progress.display(i + 1)

    return

def validate(val_loader, model, criterion, device, config):

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    accs = AverageMeter('Acc', ':6.2f', Summary.AVERAGE)
    
    progress = ProgressMeter(
        len(val_loader) + (config.distributed and (len(val_loader.sampler) * config.world_size < len(val_loader.dataset))),
        [batch_time, losses, accs],
        prefix='Test: ')

    model.eval()
    
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i                    
                images = images.to(device)
                target = target.to(device)

                # compute output
                output = model(images)
                loss = criterion(output, target)
                
                # measure acc and record loss
                acc = accuracy(output, target)
                losses.update(loss.item(), images.size(0))
                accs.update(acc, images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % config.PRINT_FREQ == 0:
                    progress.display(i + 1)

    run_validate(val_loader)
    progress.display_summary()
    return accs.avg


def parse_args():
    parser = argparse.ArgumentParser(description='Train SVTR Text Recognition Model.')
    parser.add_argument('--cfg', help='configuration file name.', required=True, type=str)
    parser.add_argument('--freq', help='frequency of logging.', default=10, type=int)
    parser.add_argument('--gpu', help='gpus', type=str)
    parser.add_argument('--worker', help='num of dataloader workers', type=int)
    
    args = parser.parse_args()
    return args

def main():
    random.seed(23)
    torch.manual_seed(23)
    np.random.seed(23)
    
    # parse configuration
    args = parse_args()
    with open(args.cfg, 'r') as f:
        config = yaml.safe_load(f)
        config = edict(config)

    # cudnn.benchmark = config.CUDNN.BENCHMARK
    # cudnn.deterministic = config.CUDNN.DETERMINISTIC
    # cudnn.enabled = config.CUDNN.ENABLED
    cudnn.benchmark = True      # benchmark: find best conv alg
    cudnn.deterministic = False # deterministic operations
    cudnn.enabled = True

    # 
    torch.distributed.is_nccl_available()
    dist.init_process_group(backend=dist.Backend.NCCL)

    if args.gpu is not None:
        config.GPUS = args.gpu
        print("Use GPU: {} for training".format(args.gpu))

    # training device
    if config.GPUS is not None and torch.cuda.is_available():
        local_rank = torch.distributed.
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    elif torch.backends.mps.is_available():
        # (GPU for MacOS devices with Metal programming framework)
        device = torch.device("mps")    
    else:
        device = torch.device("cpu")

    # resume from a checkpoint

    # data loading
    trainset = RecDataset("", "")
    trainsampler = DistributedSampler(trainset)
    trainloader = DataLoader(
        dataset=trainset, 
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        num_workers=config.WORKERS,
        sampler=trainsampler,
        pin_memory=config.PIN_MEM
        )

    valset = RecDataset("", "")
    valsampler = DistributedSampler(valset)
    valloader = DataLoader(
        dataset=valset,
        batch_size=config.VALIDATE.BATCH_SIZE_PER_GPU,
        num_workers=config.WORKERS,
        sampler=valsampler,
        pin_memory=config.PIN_MEM
    )

    model = DistributedDataParallel(model, device_ids=[0])
    criterion = Loss
    optimizer = Optimizer

    # train
    for epoch in range(config.START_EPOCH, config.EPOCHS):
        train(trainloader, model, criterion, optimizer, epoch, device, config)
        validate(valloader, model, criterion, config)

if __name__ == '__main__':
    main()
