'''
Author: Cristiano-3 chunanluo@126.com
Date: 2023-03-20 14:24:05
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2023-04-14 16:55:10
FilePath: /SVTR/tools/train.py
Description: 
'''
# coding: utf-8
import yaml
import argparse
from easydict import EasyDict as edict

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser(description='Train SVTR Text Recognition Model.')
    parser.add_argument('--cfg', help='configuration file name.', required=True, type=str)
    parser.add_argument('--freq', help='frequency of logging.', default=10, type=int)
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    
    args = parser.parse_args()
    return args

def main():
    # parse configuration
    args = parse_args()
    with open(args.cfg, 'r') as f:
        config = yaml.safe_load(f)
        config = edict(config)

    # 
    torch.distributed.is_nccl_available()
    dist.init_process_group(backend=dist.Backend.NCCL)
    

if __name__ == '__main__':
    main()
