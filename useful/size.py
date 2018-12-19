import os
from PIL import Image
import numpy as np

dir = './1.3/'
ddir = './tmp/'


# # 统计中位数

# size = []
# for i in os.listdir(dir):
#     for j in os.listdir(dir+i):
#         size.append(np.array(Image.open(dir+i+'/'+j)).shape)

# a = np.array(size)[:,0]
# b = np.array(size)[:,1]
# a = np.sort(a)
# b = np.sort(b)
# print(a[int(len(a)/2)],b[int(len(b)/2)])

for i in os.listdir(dir):
    for j in os.listdir(dir+i):
        a = Image.open(dir+i+'/'+j)
        a = a.resize((28,28),Image.ANTIALIAS)
        a.save(ddir+j)