import numpy as np
import os
import matplotlib.pyplot as plt
import PedFunctions as pf
import simplejson

os.chdir(r'd:\14GeV')

pedestals = np.load('pedestals.npy', allow_pickle=True)
pednames = np.load('pednames.npy', allow_pickle=True)

tracker = 0
index = 0
x1 = []
y1 = []
x2 = []
y2 = []
outFile = []
for i in pedestals:
    for j in i:
        if pf.convert(index) == [0, 4, 28]:
            x2.append(pednames[tracker])
            y2.append(j)
            outFile.append(int(pednames[tracker]))
            outFile.append(str([0, 3, 28]))
            outFile.append(int(j))
        if pf.convert(index) == [0, 2, 27]:
            x1.append(pednames[tracker])
            y1.append(j)
            outFile.append(int(pednames[tracker]))
            outFile.append(str([0, 2, 27]))
            outFile.append(int(j))
        index += 1
    index = 0
    tracker += 1


plt.plot(x1, y1)
plt.title('0, 2, 27')
plt.xticks(rotation=90)
plt.xlim(0, 100)
plt.show()

plt.plot(x1, y1)
plt.title('0, 2, 27')
plt.xticks(rotation=90)
plt.xlim(100, 181)
plt.show()

plt.plot(x2, y2)
plt.title('0, 4, 28')
plt.xticks(rotation=90)
plt.xlim(0, 100)
plt.show()

plt.plot(x2, y2)
plt.title('0, 4, 28')
plt.xticks(rotation=90)
plt.xlim(100, 181)
plt.show()

outFile = np.asarray(outFile)
f = open('3and2peds.txt', 'w')
simplejson.dump(outFile.tolist(), f)
