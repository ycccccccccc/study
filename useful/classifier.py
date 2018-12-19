import os
import shutil
import random

sdir = './1.3_pre_aug/'#来源
ddir = './1.3_test/'#目的

path = os.listdir(sdir)

# for i in path:
#     sourceFile = sdir+i
#     targetDir = ddir+i[i.find('_')+1]+'/'
#     targetFile = targetDir+i
#     if not os.path.exists(targetDir):
#         os.makedirs(targetDir)
#     shutil.copyfile(sourceFile, targetFile)
#
for i in path:
    sourceDir = sdir + i+'/'+i + '/'
    targetDir = ddir + i + '/'
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
    file = os.listdir(sourceDir)
    random.shuffle(file)
    for j in file[int(0.8*len(file)):]:
        sourceFile = sourceDir + j
        targetFile = targetDir + j
        shutil.move(sourceFile, targetFile)
