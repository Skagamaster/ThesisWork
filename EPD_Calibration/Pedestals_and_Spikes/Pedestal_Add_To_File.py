import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import simplejson
import uproot as up
from matplotlib.backends.backend_pdf import PdfPages


def EPDTile(tile=int(123)):
    if tile < 372:
        ew = 0
    else:
        ew = 1
    pp = int((tile - int(tile/372)*372)/31) + 1
    tt = tile % 31 + 1
    return [ew, pp, tt]


os.chdir(r"D:\14GeV")

print("Importing data ...")

names = np.load("pednames.npy", allow_pickle=True)
peds = np.load("pedestals.npy", allow_pickle=True)

os.chdir('ChiFit')

leaveit = [[0, 1, 12], [0, 3, 1], [0, 3, 3], [0, 3, 5], [0, 6, 2], [0, 7, 1],
           [0, 9, 1], [0, 9, 26], [0, 10, 5], [1, 1, 5], [1, 3, 2], [1, 3, 20],
           [1, 7, 2], [1, 7, 16], [1, 8, 2], [1, 9, 15]]

data = np.loadtxt('Nmip_Day_123.txt')
myfile = open('Nmip_Day_333.txt', 'w')
for i in range(int(len(data))):
    if EPDTile(i) in leaveit:
        myfile.write("%d\t%d\t%d\t%d\t%f\t%f\n" % (int(data[i][0]), int(
            data[i][1]), int(data[i][2]), int(data[i][3]), data[i][4], int(data[i][5])))
        continue
    myfile.write("%d\t%d\t%d\t%d\t%f\t%f\n" % (int(data[i][0]), int(
        data[i][1]), int(data[i][2]), int(data[i][3]), data[i][4], int(-peds[12][i])))
