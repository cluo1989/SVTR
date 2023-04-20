# coding: utf-8
import cv2
import random
import numpy as np
from datasets.rec.image_tool import resize_norm_img
from datasets.rec.augmenter import ImageAugmenter
from utils.charset import alphabet as charset

# error labels
errlabel = {'㎡':'m2', 'Ⅲ':'III', 'Ⅱ':'II', '＞':'>', 'Ⅴ':'V', '＝':'=', '～':'~', 'Α':'A', '∪':'U', '㎎':'mg', \
            '³':'3', 'а':'a', 'Β':'B', 'К':'K', '＋':'+', 'Р':'p', 'Ⅸ':'IX', '㎝':'cm', '１':'1', 'Ⅰ':'|', '？':'?', \
            '＜':'<', '％':'%', '￥':'¥', '）':')', '（':'(', '；':';', '：':':', '〓':'='}

def char_to_index(word):
    """ change str to index in charset
    """
    indexs = []
    for c in word:
        try:
            if c in errlabel:
                c = errlabel[c]

            if len(c) == 1:
                idx = charset.index(c)
                indexs.append(idx)

            else:
                for cc in c:
                    idx = charset.index(cc)
                    indexs.append(idx)
                    
        except:
            #print('********* char index failed ********', word)
            return None

    return indexs

class DatasetReader(object):
    def __init__(self, params):
        print(params)
        self.mode = params['mode'] #"train" 
        self.real_ratio = 0.5#0 #params['REAL_RATIO']
        self.num_workers = 4 #params['NUM_WORKERS']
        self.image_shape = [32, 320, 1] #params['INPUT_SHAPE']
        self.max_label_len = 25 #params['MAX_LABEL_LEN']
        self.infer_img_dir = None
        self.use_distort = False
        self.mark_between_img_label = '.jpg    '
        self.augmenter = ImageAugmenter()
        
        with open(params['simu'], 'r') as fin:
            self.simu_list = fin.readlines()
        with open(params['real'], 'r') as fin:
            self.real_list = fin.readlines()

        random.shuffle(self.simu_list)
        random.shuffle(self.real_list)
    
    def read_img_list(self, dataset_dirs):
        data_list = []
        for dataset_dir in dataset_dirs:
            # label file path
            label_file_name = dataset_dir.split('/')[-1] + '.txt'
            dataset_label_file = dataset_dir + '/' + label_file_name
            
            # read and append samples
            with open(dataset_label_file, 'r') as fin:
                lines = fin.readlines()
                lines = [dataset_dir + '/' + line for line in lines]
                print(dataset_label_file, ':', len(lines))
                data_list.extend(lines)

        print('total:', len(data_list))
        return data_list

    def __call__(self):
        
        def sample_iter_reader():
            if self.mode == 'test' and self.infer_img_dir is not None:
                img_list = get_img_list(self.infer_img_dir)

            else:
                num_simu = len(self.simu_list)
                num_real = len(self.real_list)

                cnt_simu = 0
                cnt_real = 0
                mark = self.mark_between_img_label

                while True:
                    if cnt_simu == num_simu:
                        random.shuffle(self.simu_list)
                        cnt_simu = 0
                        print('shuffle simu final')
                    if cnt_real == num_real:
                        random.shuffle(self.real_list)
                        cnt_real = 0
                        print('shuffle real final')

                    p = np.random.uniform(0, 1)
                    if p >= self.real_ratio:                        
                        # simulate dataset
                        mark = ".jpg    "
                        record = self.simu_list[cnt_simu]
                        cnt_simu += 1
                        #print('simu:', cnt_simu)
                                                
                    else:                        
                        # real dataset
                        mark = ".jpg    "
                        record = self.real_list[cnt_real]
                        cnt_real += 1
                        #print('real:', cnt_real)

                    if mark not in record:
                        print(record)
                        continue

                    t = record.strip('\n').split(mark)
                    if len(t) != 2:
                        print(record)
                        continue

                    imgname, label = t[:2]
 
                    if label != ' ':
                        label = label.strip()
 
                    # check label length
                    if label == '@@@@' or len(label) > self.max_label_len:
                        continue
                    
                    imgname += '.jpg'

                    if len(label) == 0:
                        print('label length is 0:', imgname)
                        continue

                    # read
                    imgname = imgname.replace('/home/datasets/','/media/cluo/0000B445000DC220/t1/第一批/image_cut_for_labeling/')
                    img = cv2.imread(imgname)
                    if img is None:
                        print('image is None:', imgname)
                        continue
                    
                    label = char_to_index(label)
                    if label is None:
                        print('label encode failed:', imgname)
                        continue

                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                    # augment simu images
                    if p >= self.real_ratio:
                        img = self.augmenter.image_augmentation(img)

                    #==============================================
                    if np.random.uniform(0, 1) <= 0.001:
                        # 两边扩展指定长度
                        ph = 0
                        pw = np.random.randint(32)
                        pv = 0
                        hwc = list(img.shape)
                        hwc[0] += 2 * ph
                        hwc[1] += 2 * pw

                        # 填充的值
                        if pv is None:
                            pv = 0
                        elif pv == -1:
                            pv = img.mean()

                        pad_img = pv * np.ones(hwc, dtype=img.dtype)
                        
                        # left_gradual_weights = np.arange(0,1.0,1/(pw+1)).reshape((1,pw+1,1))
                        # right_gradual_weights= np.arange(1.0,0,-1/pw).reshape((1,pw,1))
                        
                        # pad_img[:, :pw+1, ...] = left_gradual_weights * np.expand_dims(img[:, 0,...], axis=1)
                        # pad_img[:, -pw:, ...] = right_gradual_weights * img[:, -1:,...]

                        # pad_img[:, :pw, ...] = img[:, :2, ...].mean()
                        # pad_img[:, -pw:, ...] = img[:, -2:, ...].mean()
                        pad_img[ph:hwc[0]-ph, pw:hwc[1]-pw, ...] = img
                        img = pad_img
                    #==============================================

                    # resize and normalization
                    img = resize_norm_img(img, self.image_shape)

                    # pad label
                    if len(label) > self.max_label_len:
                        continue

                    label = np.array(label)

                    pad_label = np.zeros(self.max_label_len) #8275*np.ones(self.max_label_len)
                    pad_label[:len(label)] = label

                    # get length
                    label_len = [len(label)]

                    yield img, pad_label, label_len

        return sample_iter_reader()


if __name__ == "__main__":
    reader = DatasetReader([])
    gen = reader(0)()
    for i in range(10):
        img, label = next(gen)
        print(img.shape, label)
