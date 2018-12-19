from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from PIL import Image
import matplotlib as pyplot
import numpy as np
import scipy
import os
import cv2 as cv
from keras.preprocessing import image

# RANGE_DIR = [0,1,2,3,4,5,6,7,8,9,"A","C","D","E","F","K","L","M","R","S","T","U","V","X","Y"]
RANGE_DIR = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'S', 'E', 'P', 'H', 'M', 'B', 'D', 'J', 'I', 'X', 'Z', 'T', 'A', 'K', 'V',
             'U', 'Y', 'C', 'L','F', 'R']
shift = 0.2
# width_shift_range=shift,height_shift_range=shift
# rotation_range = 20
# channel_shift_range = 100
# zoom_range = 0.5
# shear_range = 0.5
datagen = image.ImageDataGenerator(shear_range = 0.5)
for i in RANGE_DIR:
    dir = "./1.3_pre_aug/%s/" % i
    sdir = "./tmp/%s/" % i
    len1 = len(os.listdir(dir+str(i)))
    print(len1)
    if not os.path.exists(sdir):
        os.makedirs(sdir)
    gen_data = datagen.flow_from_directory(dir, batch_size=1, shuffle=False, save_to_dir=sdir,
                                           save_prefix='shear', target_size=(32, 32))
    for j in range(len1):
        gen_data.next()
