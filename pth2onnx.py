import os
import yaml
import torch
import argparse
from easydict import EasyDict as edict
from PIL import Image
import numpy as np

from modeling.architecture.rec_model import RecModel
from datasets.image_tool import resize_norm_img
from modeling.postprocess.rec_postprocess import CTCDecode


def parse_args():
    parser = argparse.ArgumentParser(description='Train SVTR Text Recognition Model.')
    parser.add_argument('-c', '--cfg', help='configuration file name.', required=True, type=str)
    parser.add_argument('-m', '--model_file', help="path of model file.", required=True, type=str)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    with open(args.cfg, 'r') as f:
        config = yaml.safe_load(f)
        config = edict(config)

    checkpoint = torch.load(args.model_file)
    model = RecModel(config['MODEL'])
    model.to('cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    

    X_test = torch.rand(1,1,32,320)
    torch.onnx.export(model, X_test, 'ocr.onnx', input_names=["inputs"], output_names=["fc"])
    

if __name__ == '__main__':
    main()
