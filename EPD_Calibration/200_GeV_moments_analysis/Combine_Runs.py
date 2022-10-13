# This is just to get all picos into a "by run" format.

import os
import numpy as np

os.chdir(r'F:\AuAu200')
files = []
for i in os.listdir():
    if i.endswith('root'):
        files.append(i)
print(len(files))

runs = []
count1 = 0
count2 = 0
for i in files:
    if i[11] == '2':
        run = int(i[11:19])
    else:
        run = int(i[15:23])
    runs.append(run)
print(np.unique(runs))
