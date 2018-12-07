import numpy as np
import os
from PIL import Image

dir = '../../data/number/'
RANGE_DIR = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
             "L", "M", "P", "R", "S", "U", "V", "X", "Y", "Z"]

train_image_path = []
train_image_lable = []
test_image_path = []
test_image_lable = []
path = os.listdir(dir)
class_num = 10
image_size = [224, 224, 3]

# 提取训练集图片地址和lable
for i in os.listdir(dir):
    sub_path = os.listdir(dir + i)
    np.random.shuffle(sub_path)
    for j in sub_path[:int(0.7 * len(sub_path))]:
        train_image_path.append(dir + i + '/' + j)
        train_image_lable.append(RANGE_DIR.index(i))
    for j in sub_path[int(0.7 * len(sub_path)):]:
        test_image_path.append(dir + i + '/' + j)
        test_image_lable.append(RANGE_DIR.index(i))

num_examples = len(train_image_lable)
test_number = len(test_image_lable)


def parse_image(path):
    img = np.array(Image.open(path)).reshape(image_size[0] * image_size[1] * image_size[2])
    return img / 255.


def onehot(num):
    a = np.zeros(class_num, dtype=int)
    a[num] = 1
    return a


def batch(size):
    tmp_image = np.zeros([size, image_size[0] * image_size[1] * image_size[2]])
    tmp_lable = np.zeros([size, class_num])
    index = np.random.randint(0, len(train_image_lable), size)
    for i in range(size):
        tmp_image[i] = parse_image(train_image_path[index[i]])
        tmp_lable[i] = onehot(int(train_image_lable[index[i]]))
    return tmp_image, tmp_lable


def test(size=0):
    if size == 0:
        num = len(test_image_lable)
    else:
        num = size
    index = np.random.randint(0, len(test_image_lable), num)
    tmp_image = np.zeros((num, image_size[0] * image_size[1] * image_size[2]))
    tmp_lable = np.zeros((num, class_num))
    for i in range(num):
        tmp_image[i] = parse_image(test_image_path[i])
        tmp_lable[i] = onehot(int(test_image_lable[i]))
    if size != 0:
        for i in range(num):
            tmp_image[i] = parse_image(test_image_path[index[i]])
            tmp_lable[i] = onehot(int(test_image_lable[index[i]]))
    return tmp_image, tmp_lable
