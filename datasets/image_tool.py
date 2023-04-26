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