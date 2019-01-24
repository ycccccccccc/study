import os
import shutil
import random

sdir = './tmp_classed'
train_dir = './tmp_train'
test_dir = './tmp_test'

for i in os.listdir(sdir):
    path = os.path.join(sdir,i)
    file_list = os.listdir(path)
    random.shuffle(file_list)
    index = int(0.7 * len(file_list))
    for j in file_list[:index]:
        file = os.path.join(path,j)
        tar_path = os.path.join(train_dir,i)
        if not os.path.exists(tar_path):
            os.mkdir(tar_path)
        tar_file = os.path.join(tar_path,j)
        shutil.copy(file,tar_file)
    for j in file_list[index:]:
        file = os.path.join(path, j)
        tar_path = os.path.join(test_dir, i)
        if not os.path.exists(tar_path):
            os.mkdir(tar_path)
        tar_file = os.path.join(tar_path, j)
        shutil.copy(file, tar_file)


