import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm
import uproot as up
import os
from matplotlib.colors import LogNorm
from numba import vectorize

os.chdir(r'D:\19GeV\Picos')
#data = up.open('st_mtd_19142034_raw_1000002.picoDst.root')
#PicoDst = data["PicoDst"]
ADC = []
for k in range(2):
    ew = []
    for j in range(12):
        pp = []
        for i in range(31):
            tt = []
            pp.append(tt)
        ew.append(pp)
    ADC.append(ew)


def make_adc(PicoDst, ADC):

    EpdId = PicoDst.arrays([b"EpdHit.mId"])[b"EpdHit.mId"][:]
    EpdQTdata = PicoDst.arrays([b"EpdHit.mQTdata"])[b"EpdHit.mQTdata"][:]

    print("Number of events to read:", len(EpdId))

    for i in range(len(EpdId)):
        for j in range(len(EpdQTdata[i])):
            if EpdId[i][j] < 0:
                ew = 0
            else:
                ew = 1
            pp = int(str(abs(EpdId[i][j]))[:-2])-1
            tt = int(str(EpdId[i][j])[-2:])-1
            adc = int(bin(EpdQTdata[i][j])[-12:], 2)
            ADC[ew][pp][tt].append(adc)
    return ADC


for day in range(88, 94):
    os.chdir("Day%s" % day)
    R = 1
    L = len(os.listdir())
    for filename in os.listdir():
        print("File", R, "of", L)
        data = up.open(filename)
        if not "PicoDst" in data:
            R += 1
            continue
        PicoDst = data["PicoDst"]
        ADC = make_adc(PicoDst, ADC)
        R += 1
    os.chdir(r'D:\19GeV\Picos')
    print("Saving ...")
    np.save("Day%sarray" % day, ADC)
    print("Saved as Day%sarray." % day)
    print("Entries on 1st tile:", len(ADC[0][0][0]))

# getArray = np.load('Day142array.npy', allow_pickles=True)
