import numpy as np
import os
from PIL import Image

train_dir = './train_299/'
# dir = './data/data_all_ex/test/'
test_dir = './test/'

RANGE_DIR = ["A","B","C","D","E","F","G","H","I","J","K","L","M","0","P","R","S","T","U","V","X","Y","Z"]

train_image_path = []
train_image_lable = []
test_image_path = []
test_image_lable = []
classnum = len(RANGE_DIR)
image_size = [299,299,3]

#提取训练集图片地址和lable
for i in os.listdir(train_dir):
    if i in RANGE_DIR:
        sub_path = os.listdir(train_dir+i)
        for j in sub_path:
            train_image_path.append(train_dir+i+'/'+j)
            train_image_lable.append(RANGE_DIR.index(i))

for i in os.listdir(test_dir):
    if i in RANGE_DIR:
        sub_path = os.listdir(test_dir+i)
        for j in sub_path:
            test_image_path.append(test_dir+i+'/'+j)
            test_image_lable.append(RANGE_DIR.index(i))

def parse_image(path):
    img = np.array(Image.open(path))
    return img/255.

def onehot(num):
    a = np.zeros(classnum,dtype=int)
    a[num] = 1
    return a

def batch(size):
    tmp_image = np.zeros([size,299,299,3])
    tmp_lable = np.zeros([size,classnum])
    index = np.random.randint(0,len(train_image_lable),size)
    for i in range(size):
        tmp_image[i] = parse_image(train_image_path[index[i]])
        tmp_lable[i] = onehot(int(train_image_lable[index[i]]))
    return tmp_image,tmp_lable

def test(size = 0):
    if size == 0:
        size = len(test_image_lable)
    tmp_image = np.zeros([size,299,299,3])
    tmp_lable = np.zeros([size,classnum])
    index = np.random.randint(0,len(test_image_lable),size)
    for i in range(size):
        tmp_image[i] = parse_image(test_image_path[index[i]])
        tmp_lable[i] = onehot(int(test_image_lable[index[i]]))
    return tmp_image, tmp_lable
