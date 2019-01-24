# from keras.preprocessing.image import ImageDataGenerator
# from keras.datasets import mnist
# from PIL import Image
# import matplotlib as pyplot
# import numpy as np
# import scipy
import os
import shutil
# import cv2 as cv
from keras.preprocessing import image

ddir = './tmp_train/'
ssdir = './tmp_aug/'

RANGE_DIR = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'S', 'E', 'P', 'H', 'M', 'B', 'D', 'J', 'I', 'X', 'Z', 'T', 'A', 'K', 'V',
             'U', 'Y', 'C', 'L', 'F', 'R']


shift = 0.2
arg = {"shift": {"width_shift_range": shift, "height_shift_range": shift},
       "rotate": {"rotation_range": 20},
       "channel": {"channel_shift_range": 100},
       "zoom": {"zoom_range": 0.5},
       "shear": {"shear_range": 0.5}}

def mkd():
    for i in os.listdir(ddir):
        path = os.path.join(ddir,i)
        os.mkdir(os.path.join(path,i))
        for j in os.listdir(path):
            file = os.path.join(path,j)
            if os.path.isfile(file):
                shutil.move(file,os.path.join(path,i,j))

def aug(method):
    datagen = image.ImageDataGenerator(**arg[method])
    for i in os.listdir(ddir):
        dir = os.path.join(ddir,i)
        sdir = os.path.join(ssdir,i)
        len1 = len(os.listdir(os.path.join(dir,i)))
        print(len1)
        if not os.path.exists(sdir):
            os.makedirs(sdir)
        gen_data = datagen.flow_from_directory(dir, batch_size=1, shuffle=False, save_to_dir=sdir,
                                               save_prefix=method, target_size=(32, 32))
        for j in range(len1):
            gen_data.next()

mkd()

for i in arg:
    aug(i)