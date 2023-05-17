# coding: utf-8
import math
import numpy as np
from PIL import Image


def resize_norm_img(img, image_shape):
    imgH, imgW, imgC = image_shape

    h = img.shape[0]
    w = img.shape[1]
    ratio = w / float(h)

    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    
    # resize
    img = Image.fromarray(img)
    resized_image = img.resize(size=(resized_w, imgH))
    resized_image = np.asarray(resized_image)

    # normalize
    resized_image = resized_image.astype('float32')
    resized_image = resized_image / 255
    resized_image -= 0.5
    resized_image /= 0.5

    # padding
    padding_im = np.zeros((imgH, imgW, imgC), dtype=np.float32)
    padding_im[:, 0:resized_w, ...] = np.expand_dims(resized_image, -1)
    return padding_im

def random_pad(img):
    # 0.1% randomly pad black borders
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
        pad_img[ph:hwc[0]-ph, pw:hwc[1]-pw, ...] = img
        img = pad_img
    
    return img
