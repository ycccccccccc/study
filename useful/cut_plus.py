import os
import numpy as np
from PIL import Image

sdir = './cut_100/'#来源
ddir = './tmp/'#目的

list = os.listdir(sdir)

count = 0
for i in list:
    if 'jpg' in i:
        index = i.find('(')
        if index == -1:
            index = i.find('.')
        img = np.array(Image.open(sdir+i))
        len = img.shape[1]
        aaaa = 3/20.0
        ex = int(len / index * aaaa)
        for j in range(index):
            tmp = img[:,int(max(0,len * j/index-ex)):int(min(len,len * (j+1)/index+ex))]
            if j==0:
                tmp = np.concatenate(((img[:,0:ex]),tmp),axis=1)
            if j==index-1:
                tmp = np.concatenate(((tmp,img[:,len-ex:len])),axis=1)

            tmp = Image.fromarray(tmp)
            tmp.save(ddir+i[:-4]+'_'+i[j]+str(j)+'.jpg')
            count += 1
print(count)