import os
import shutil
import numpy as np

path = r'F:\2019Picos\14p5GeV\Runs'
goodruns = np.load(r'D:\14GeV\Thesis\goodruns.npy', allow_pickle=True)
baddruns = np.load(r'D:\14GeV\Thesis\badruns.npy', allow_pickle=True)
files = []
size = 0
count = 0
print("Working on file:")
for i in os.listdir(path):
    if int(i[:8]) <= 20132007:
        continue
    if size >= 500000000:
        break
    if int(i[:8]) in goodruns:
        files.append(i)
        count += 1
        size += os.path.getsize(path + '\\' + i)/1024
        shutil.copy(path + '\\' + i,
                    r'D:\14GeV\Full_Picos' + '\\' + i)
print(size/1000, "MB transferred.")
print(files[-1])
