import os
import numpy as np
dir = './tmp/'
list = os.listdir(dir)

num = np.zeros(255)
for sub_dir in list:
    sub_list =os.listdir(dir + sub_dir)
    num[ord(sub_dir)] += len(sub_list)

for i in range(255):
    if num[i] != 0:
        print(chr(int(i))+'\t',int(num[i]))