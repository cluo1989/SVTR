'''
Author: Cristiano-3 chunanluo@126.com
Date: 2023-04-17 10:47:24
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2023-04-26 16:52:36
FilePath: /SVTR/datasets/mydataset.py
Description: 
'''
import torch
from torch import nn
from torch.utils import data
from PIL import Image
import numpy as np
from datasets.label_converter import encode
from datasets.image_tool import resize_norm_img


class RecDataset(data.Dataset):
    def __init__(self, label_file, image_dir):
        super().__init__()
        self.image_shape = [32, 320, 1]
        self.max_label_len = 25
        self.split_mark = '    '
        self.old_image_dir = '/home/datasets/'
        self.samples = self.load_labels(label_file, image_dir)[:100]

    def debug_print(self, tip, content, debug=False):
        if debug:
            print(tip, content)

    def load_labels(self, label_file, image_dir):
        samples = []
        with open(label_file, 'r') as fin:
            lines = fin.readlines()
            for line in lines:
                if self.split_mark not in line:
                    self.debug_print('mark not found:', line)
                    continue
                
                tmp = line.strip().split(self.split_mark)
                if len(tmp) != 2:
                    self.debug_print('len(tmp)!=2:', line)
                    continue

                image_name, label = tmp[:2]
                label = label.strip()

                if len(label) == 0:
                    self.debug_print('len(label)==0:', label)
                    continue

                if label == '@@@@':
                    self.debug_print('@@@@:', label)
                    continue

                if len(label) > self.max_label_len:
                    self.debug_print('long label(>25):', label)
                    continue

                label = encode(label)
                if label is None:
                    self.debug_print('encode failed:', label)
                    continue

                image_name = image_name.replace(self.old_image_dir, '')
                image_file = '/'.join([image_dir, image_name])
                samples.append((image_file, label))
        return samples
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_file, label = self.samples[index]
        image = Image.open(image_file)

        # resize and normalize image
        image = np.asarray(image)
        image = resize_norm_img(image, self.image_shape)
        image = torch.from_numpy(image).float()
        image = image.permute([2,0,1])
        
        # pad label
        label = np.array(label, dtype=np.int32)
        pad_label = np.zeros(self.max_label_len, dtype=np.int32) #8275*np.ones(self.max_label_len)
        pad_label[:len(label)] = label
        
        label_len = len(label)#np.array([])
        return image, pad_label, label_len


if __name__ == '__main__':
    testset = RecDataset('/home/cluo/Workspace/Projects/general_ocr/datasets/rec/datas/test_real.txt', \
                         '/media/cluo/0000B445000DC2201/t1/第一批/image_cut_for_labeling/')

    testloader = data.DataLoader(testset, batch_size=1, shuffle=True)
    image, label = next(iter(testloader))
    print(type(image), type(label))
    print(image.shape, label)
