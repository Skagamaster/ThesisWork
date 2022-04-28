import numpy as np
import math
import uproot as up
import os

os.chdir(r'D:\19GeV\ChiFit')

nMIP = np.zeros([2, 12, 31])
calibrate = np.loadtxt('Nmip_Day_88.txt')
for i in range(2):
    for j in range(12):
        for k in range(31):
            nMIP[i][j][k] = calibrate[31*12*i + 31*j + k][4]

os.chdir(r'D:\19GeV\Picos\Raw')


def fill_array(PicoDst, array):

    # These arrays are for the data out of the .root file.
    EpdId = PicoDst.arrays([b"EpdHit.mId"])[b"EpdHit.mId"][:]
    # mEpdnMIP = PicoDst.arrays([b"EpdHit.mnMIP"])[b"EpdHit.mnMIP"][:]
    RefMult = PicoDst.arrays([b"Event.mGRefMult"])[b"Event.mGRefMult"][:]
    xVert = PicoDst.arrays([b"Event.mPrimaryVertexX"])[
        b"Event.mPrimaryVertexX"][:]
    yVert = PicoDst.arrays([b"Event.mPrimaryVertexY"])[
        b"Event.mPrimaryVertexY"][:]
    zVert = PicoDst.arrays([b"Event.mPrimaryVertexZ"])[
        b"Event.mPrimaryVertexZ"][:]
    VzVpd = PicoDst.arrays([b"Event.mVzVpd"])[b"Event.mVzVpd"][:]
    NBTOFMatch = PicoDst.arrays([b"Event.mNBTOFMatch"])[
        b"Event.mNBTOFMatch"][:]
    EpdQTdata = PicoDst.arrays([b"EpdHit.mQTdata"])[b"EpdHit.mQTdata"][:]

    # This is the fill with event cuts. The cuts are as follows:
    # |Vz| < 40.0, |Vr| < 2.0
    for i in range(len(RefMult)):
        if abs(zVert[i][0]) < 40.0 and math.sqrt(xVert[i][0]**2+yVert[i][0]**2) < 2.0 \
                and NBTOFMatch[i][0] > 0.00 and abs(zVert[i][0]-VzVpd[i][0]) <= 10.0:
            y = []
            x = np.zeros(34)
            x[0] += RefMult[i][0]
            x[1] += zVert[i][0]
            for j in range(len(EpdId[i])):
                if EpdId[i][j] < 0:
                    ew = 0
                else:
                    ew = 1
                pp = int(str(abs(EpdId[i][j]))[:-2])-1
                tt = int(str(EpdId[i][j])[-2:])-1
                adc = int(bin(EpdQTdata[i][j])[-12:], 2)
                val = int(str(EpdId[i][j])[-2:])
                val = (val - val % 2)/2
                absval = abs(EpdId[i][j])/EpdId[i][j]
                point = int(val+2+8*(1+absval))
                x[point] += adc/nMIP[ew][pp][tt]
            for t in range(34):
                y.append(x[t])
            array.append(y)
    return array


array = []
R = 1
L = len(os.listdir())
for PicoFile in os.listdir():
    print("Working on file", R, "of", L)
    data = up.open(PicoFile)
    if not "PicoDst" in data:
        R += 1
        continue
    PicoDst = data["PicoDst"]
    R += 1
    array = fill_array(PicoDst, array)

array = np.asarray(array)
os.chdir(r'D:\19GeV\data')
np.savetxt("array.txt", array, delimiter=' ')
