import numpy as np
import math
import uproot as up
import os

os.chdir(r'D:\19GeV\Picos\Raw')

# This is the function I'll use to create the nMIP arrays.


def fill_array(PicoDst, array):

    # These jagged arrays are to get the data out of the .root file.
    mEpdId = PicoDst.arrays([b"EpdHit.mId"])
    mEpdnMIP = PicoDst.arrays([b"EpdHit.mnMIP"])
    mRefMult = PicoDst.arrays([b"Event.mGRefMult"])
    mxVert = PicoDst.arrays([b"Event.mPrimaryVertexX"])
    myVert = PicoDst.arrays([b"Event.mPrimaryVertexY"])
    mzVert = PicoDst.arrays([b"Event.mPrimaryVertexZ"])
    mVzVpd = PicoDst.arrays([b"Event.mVzVpd"])
    mNBTOFMatch = PicoDst.arrays([b"Event.mNBTOFMatch"])

    # These are flattened arrays, and all are ordered by event.
    EpdId = mEpdId[b"EpdHit.mId"][:]
    EpdnMIP = mEpdnMIP[b"EpdHit.mnMIP"][:]
    RefMult = mRefMult[b"Event.mGRefMult"][:]
    xVert = mxVert[b"Event.mPrimaryVertexX"][:]
    yVert = myVert[b"Event.mPrimaryVertexY"][:]
    zVert = mzVert[b"Event.mPrimaryVertexZ"][:]
    VzVpd = mVzVpd[b"Event.mVzVpd"][:]
    NBTOFMatch = mNBTOFMatch[b"Event.mNBTOFMatch"][:]

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
                val = int(str(EpdId[i][j])[-2:])
                val = (val - val % 2)/2
                absval = abs(EpdId[i][j])/EpdId[i][j]
                point = int(val+2+8*(1+absval))
                if EpdnMIP[i][j] > 3.0:
                    x[point] += 3.0
                else:
                    x[point] += EpdnMIP[i][j]
            for t in range(34):
                y.append(x[t])
            array.append(y)
    return array


# Here I fill the arrays for all files in the folder.
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

# Now I save the arrays for processing elsewhere.
array = np.asarray(array)
np.savetxt("arrayCutoff.txt", array, delimiter=' ')
