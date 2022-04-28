import numpy as np
import math
import uproot as up
import os
import matplotlib.pyplot as plt
import pandas as pd
import definitions as dfn

os.chdir(r'D:\27gev_production')
data = up.open('st_physics_19999_raw_1234.picoDst.root')

PicoDst = data["PicoDst"]

# These are flattened arrays, and all are ordered by event.
EPDid = PicoDst.arrays([b"EpdHit.mId"])[b"EpdHit.mId"][:]
nMIP = PicoDst.arrays([b"EpdHit.mnMIP"])[b"EpdHit.mnMIP"][:]
GRefMult = PicoDst.arrays([b"Event.mGRefMult"])[
    b"Event.mGRefMult"][:].flatten()
RefMult1 = PicoDst.arrays([b"Event.mRefMultPos"])[b"Event.mRefMultPos"][:].flatten(
) + PicoDst.arrays([b"Event.mRefMultNeg"])[b"Event.mRefMultNeg"][:].flatten()
xVert = PicoDst.arrays([b"Event.mPrimaryVertexX"])[
    b"Event.mPrimaryVertexX"][:].flatten()
yVert = PicoDst.arrays([b"Event.mPrimaryVertexY"])[
    b"Event.mPrimaryVertexY"][:].flatten()
zVert = PicoDst.arrays([b"Event.mPrimaryVertexZ"])[
    b"Event.mPrimaryVertexZ"][:].flatten()
VzVpd = PicoDst.arrays([b"Event.mVzVpd"])[b"Event.mVzVpd"][:].flatten()
NBTOFMatch = PicoDst.arrays([b"Event.mNBTOFMatch"])[
    b"Event.mNBTOFMatch"][:].flatten()

counter = 0
nRings = []
for i in range(int(len(EPDid))):
    if counter % 1000 == 0:
        print("working on ", counter, " of ", int(len(EPDid)))
    x = int(len(EPDid[i]))
    nRings.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for j in range(x):
        r = int(dfn.EPDRing(EPDid[i][j])[1])
        t = nMIP[i][j]
        if t < 0.3:
            t = 0
        if t > 2.0:
            t = 2.0
        nRings[i][r-1] += t
    counter += 1
nRings = np.asarray(nRings)

# Save it all as a Pandas dataframe.
rows = np.linspace(0, len(EPDid)-1, len(EPDid))
columns = ['GRefMult', 'RefMult1', 'xVert',
           'yVert', 'zVert', 'VzVPD', 'nBTOFmatch']
dArray = np.array([GRefMult, RefMult1, xVert, yVert, zVert, VzVpd, NBTOFMatch])
df = pd.DataFrame(data=dArray.T, index=rows, columns=columns)
df.to_pickle('pandasdata.pkl')
np.save('rings.npy', nRings)
