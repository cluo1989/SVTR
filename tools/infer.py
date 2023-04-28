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
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    imgdir = './images'
    imglist = os.listdir(imgdir)
    for imgfile in imglist:
        img = Image.open('/'.join([imgdir,imgfile])).convert('L')
        img = np.asarray(img)
        print(imgfile, img.shape,'------')
        h, w = img.shape[:2]
        h_ = 32
        w_ = int(w*(32./h))
        img = resize_norm_img(img, [h_, w_, 1])
        img = np.expand_dims(img, axis=0)

        img = torch.from_numpy(img).float()
        img = img.permute([0,3,1,2])
        feat = model(img).softmax(dim=2)
        
        # parse & decode
        probs, preds = torch.max(feat, dim=2)
        for prob, pred in zip(probs, preds):
            text = CTCDecode().decode(pred.detach().numpy(), prob.detach().numpy())
            print(text)
    


if __name__ == '__main__':
    main()
