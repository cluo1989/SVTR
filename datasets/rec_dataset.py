'''
Author: Cristiano-3 chunanluo@126.com
Date: 2023-04-17 10:47:24
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2023-04-17 15:02:16
FilePath: /SVTR/datasets/mydataset.py
Description: 
'''
import torch
from torch import nn
from torch.utils import data
from PIL import Image
import numpy as np


class RecDataset(data.Dataset):
    def __init__(self, label_file, image_dir):
        super().__init__()
        self.samples = self.load_labels(label_file, image_dir)


    def load_labels(self, label_file, image_dir):
        samples = []
        with open(label_file, 'r') as fin:
            lines = fin.readlines()
            for line in lines:
                image_name, label = line.strip().split('    ')
                image_name = image_name.replace('/home/datasets/', '')
                image_file = '/'.join([image_dir, image_name])
                samples.append((image_file, label))
        return samples
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_file, label = self.samples[index]
        image = Image.open(image_file)

        # image & label prepros
        image = np.asarray(image)
        image = torch.from_numpy(image).float()

        # enhancements
        
        return image, label


if __name__ == '__main__':
    testset = RecDataset('/home/cluo/Workspace/Projects/general_ocr/datasets/rec/datas/test_real.txt', \
                         '/media/cluo/0000B445000DC2201/t1/第一批/image_cut_for_labeling/')

    testloader = data.DataLoader(testset, batch_size=1, shuffle=True)
    image, label = next(iter(testloader))
    print(type(image), type(label))
    print(image.shape, label)
