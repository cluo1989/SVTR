'''
Author: Cristiano-3 chunanluo@126.com
Date: 2023-04-17 10:47:24
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2023-04-27 19:58:15
FilePath: /SVTR/datasets/mydataset.py
Description: 
'''
import torch
from torch.utils import data
from PIL import Image
import numpy as np
import random
from datasets.label_converter import encode
from datasets.image_tool import resize_norm_img


class RecDataset(data.Dataset):
    def __init__(self, 
                 real_label_file, 
                 real_image_dir, 
                 simu_label_file=None, 
                 simu_image_dir=None,
                 name=''):
        super().__init__()
        self.image_shape = [32, 320, 1]
        self.max_label_len = 20 # 25
        self.delimiter = '    '
        self.old_image_dir = '/home/datasets/'
        self.real_samples = self.load_labels(real_label_file, real_image_dir)[:64]
        # random.shuffle(self.real_samples)
        # print('shuffle real samples, init')

        if simu_label_file is not None and simu_image_dir is not None:
            self.simu_samples = self.load_labels(simu_label_file, simu_image_dir)[:128]
            # random.shuffle(self.simu_samples)
            # print('shuffle simu samples, init')
            self.real_ratio = 0.5
            self.simu_ratio = 1 - self.real_ratio
        else:
            self.simu_samples = []
            self.real_ratio = 1.0
            self.simu_ratio = 0

        # compute total according to ratio
        len_real = len(self.real_samples)
        len_simu = len(self.simu_samples)
        
        total_real = int(len_real / (self.real_ratio + 1e-6)) + 1 
        total_simu = int(len_simu / (self.simu_ratio + 1e-6)) + 1

        if total_real > total_simu:
            self.real_num = len_real
            self.simu_num = total_real - self.real_num
            self.simu_samples = random.choices(self.simu_samples, k=self.simu_num)
        else:
            self.simu_num = len_simu
            self.real_num = total_simu - self.simu_num
            self.real_samples = random.choices(self.real_samples, k=self.real_num)

        # total 
        self.total_samples = self.simu_samples + self.real_samples
        self.total_num = len(self.total_samples)

        # init shuffle
        random.shuffle(self.total_samples)

        print(f'{name} DATASET STATISTICS\n', 50*'-')
        print(f'total_num:{self.total_num}, real_num:{self.real_num}, simu_num:{self.simu_num}\n')
        #print('\n'.join([t[0] for t in self.total_samples]))

    def debug_print(self, tip, content, debug=False):
        if debug:
            print(tip, content)

    def load_labels(self, label_file, image_dir):
        samples = []
        with open(label_file, 'r') as fin:
            lines = fin.readlines()
            for line in lines:
                if self.delimiter not in line:
                    self.debug_print('mark not found:', line)
                    continue
                
                tmp = line.strip().split(self.delimiter)
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

                label = encode(label)
                if label is None:
                    self.debug_print('encode failed:', label)
                    continue

                if len(label) > self.max_label_len:
                    self.debug_print('long label(>25):', label)
                    continue
                
                image_name = image_name.replace(self.old_image_dir, '')
                image_file = '/'.join([image_dir, image_name])
                samples.append((image_file, label))
        return samples
        
    def __len__(self):
        # return len(self.samples)
        return self.total_num

    def __getitem__(self, index):
        # load sample
        image_file, label = self.total_samples[index]
        image = Image.open(image_file).convert('L')

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
