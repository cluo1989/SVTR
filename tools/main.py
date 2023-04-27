'''
Author: Cristiano-3 chunanluo@126.com
Date: 2023-03-20 14:24:05
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2023-04-27 10:58:02
FilePath: /SVTR/tools/train.py
Description: 
'''
# coding: utf-8
import os
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

from modeling.architecture.rec_model import RecModel
from modeling.loss.rec_ctc_loss import CTCLoss
from datasets.rec_dataset import RecDataset
from utils import AverageMeter, ProgressMeter, Summary
from modeling.metrics.rec_metric import RecMetric
accuracy = RecMetric()

def train(train_loader, model, criterion, optimizer, scheduler, epoch, device, config):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accs = AverageMeter('Acc', ':6.2f')
    
    progress = ProgressMeter(
        len(train_loader), 
        [batch_time, data_time, losses, accs], 
        prefix='Epoch: [{}]'.format(epoch)
        )
    
    # train mode
    model.train()
    num_batches = len(train_loader)

    end = time.time()
    for i, (images, labels, label_lengths) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        label_lengths = label_lengths.to(device, non_blocking=True)

        # compute output
        output = model(images)
        # print(20*'+', output.shape, labels.shape, label_lengths.shape)
        loss = criterion(output, labels, label_lengths)

        # measure acc and record loss
        acc = accuracy(output, labels)['acc']
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

        # save checkpoint
        step = epoch * num_batches + i + 1
        if step % config.SAVE_STEP_INTER == 0:
            save_file = os.path.join(
                config.OUTPUT_DIR, ''.join([
                f"checkpoint_{epoch}_{i}",
                f"_{round(loss.item(), 4)}",
                f"_{round(acc, 4)}.pth"])
                )

            torch.save({
                'epoch': epoch,
                'step': step,
                'best_acc': acc,
                'loss': loss,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),                
                'scheduler': scheduler.state_dict()
            }, save_file)

def validate(val_loader, model, criterion, device, config):

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    accs = AverageMeter('Acc', ':6.2f', Summary.AVERAGE)
    
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, accs],
        prefix='Test: ')

    model.eval()
    
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, labels, label_lengths) in enumerate(loader):
                i = base_progress + i                    
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                label_lengths = label_lengths.to(device, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, labels, label_lengths)
                
                # measure acc and record loss
                acc = accuracy(output, labels)['acc']
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
    parser.add_argument('-c', '--cfg', help='configuration file name.', required=True, type=str)
    parser.add_argument("--local_rank", help="local device id on current node", type=int)
    # parser.add_argument('--freq', help='frequency of logging.', default=10, type=int)
    # parser.add_argument('--gpu', help='gpus', type=str)
    # parser.add_argument('--worker', help='num of dataloader workers', type=int)
    
    args = parser.parse_args()
    return args

def main():    
    # parse configuration
    args = parse_args()
    with open(args.cfg, 'r') as f:
        config = yaml.safe_load(f)
        config = edict(config)

    # set seeds
    if config.SEED is not None:
        random.seed(config.SEED)
        np.random.seed(config.SEED)
        torch.manual_seed(config.SEED)
        # torch.cuda.manual_seed(config.SEED)
        # torch.cuda.manual_seed_all(config.SEED)

        cudnn.benchmark = config.CUDNN.BENCHMARK
        cudnn.deterministic = config.CUDNN.DETERMINISTIC
        cudnn.enabled = config.CUDNN.ENABLED
        # cudnn.benchmark = False
        # cudnn.deterministic = True
        # cudnn.enabled = False
    else:
        cudnn.benchmark = config.CUDNN.BENCHMARK
        cudnn.deterministic = config.CUDNN.DETERMINISTIC
        cudnn.enabled = config.CUDNN.ENABLED
        # cudnn.benchmark = True      # benchmark: find best conv alg
        # cudnn.deterministic = False # deterministic operations
        # cudnn.enabled = True

    # training device
    if torch.cuda.is_available():
        n_gpus = 1
        dist.init_process_group(backend=dist.Backend.NCCL, world_size=n_gpus, rank=args.local_rank)

        # local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)  # f'cuda:{args.local_rank}'

    elif torch.backends.mps.is_available():
        # GPU for MacOS devices
        device = torch.device("mps")
    else:
        dist.init_process_group(backend=dist.Backend.GLOO, world_size=1, rank=args.local_rank)
        device = torch.device("cpu")
    
    # create & load model
    model = RecModel(config['MODEL'])
    criterion = CTCLoss().to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-4)

    # lr_scheduler ref: https://zhuanlan.zhihu.com/p/352744991
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        config.LR_STEP,
        config.LR_FACTOR,
        last_epoch=config.START_EPOCH - 1  # START_EPOCH = 0,1,2...
    )
    

    # resume
    best_acc = 0.0
    resume_file = config.RESUME
    if resume_file:
        if os.path.isfile(resume_file):
            print(f"loading checkpoint {resume_file}")
            if torch.cuda.is_available():
                checkpoint = torch.load(resume_file, map_location=device)
            else:
                checkpoint = torch.load(resume_file)
                
            # parse & load from checkpoint
            config.START_EPOCH = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            if torch.cuda.is_available():
                best_acc.to(device)
            
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            
            print(f"=> loaded checkpoint {resume_file} (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at {resume_file}")

    model.to(device)
    model = DistributedDataParallel(
            model, 
            # device_ids=[args.local_rank], 
            # output_device=args.local_rank
            )

    # data loading
    trainset = RecDataset(config.DATASETS.train.label_file, config.DATASETS.train.image_dir)
    trainsampler = DistributedSampler(trainset)
    trainloader = DataLoader(
        dataset=trainset, 
        batch_size=config.BATCH_SIZE_PER_GPU,
        num_workers=config.WORKERS,
        sampler=trainsampler,
        pin_memory=config.PIN_MEM
        )

    valset = RecDataset(config.DATASETS.val.label_file, config.DATASETS.val.image_dir)
    valsampler = DistributedSampler(valset)
    valloader = DataLoader(
        dataset=valset,
        batch_size=config.BATCH_SIZE_PER_GPU,
        num_workers=config.WORKERS,
        sampler=valsampler,
        pin_memory=config.PIN_MEM
    )

    # train
    for epoch in range(config.START_EPOCH, config.EPOCHS):
        trainsampler.set_epoch(epoch)
        train(trainloader, model, criterion, optimizer, scheduler, epoch, device, config)
        acc = validate(valloader, model, criterion, device, config)
        scheduler.step()

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if is_best:
            save_file = os.path.join(config.OUTPUT_DIR, f'model_best_{epoch}_{best_acc}.pth')
            torch.save({
                'epoch': epoch, 
                'best_acc': best_acc,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(), 
                'scheduler': scheduler.state_dict()
            }, save_file)

if __name__ == '__main__':
    main()
