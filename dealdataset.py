import os
import random

FolderPath='../data/JPEGImages/'
# fileFolder
print(os.path.exists(FolderPath))
flist = os.listdir(FolderPath)
numtotal = len(flist)
numval = int(len(flist)*0.2)
# print(len(flist))
seq = list(range(0,numtotal))
indices = random.sample(seq,numtotal)
f1 = open('train.txt','w')
f2 = open('val.txt','w')
for i in range(0,numtotal):
    idx = indices[i]
    fname = flist[i].split('.')[0]
    if idx < numval:
        f2.write('%s\n' % fname)
    else :
        f1.write('%s\n' % fname)