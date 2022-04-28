import numpy as np
import os
import pandas as pd
import definitions as dfn
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Loading the data from Import_Data.py
os.chdir(r'D:\27gev_production')
df = pd.read_pickle('pandasdata.pkl')
rVert = (df.xVert**2+df.yVert**2)**(1/2)
df['rVert'] = rVert
rings = np.load('rings.npy', allow_pickle=True)

# Some before cut plots.
#cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
plt.figure(figsize=(8, 5))
plt.hist2d(df.xVert, df.yVert, range=[[-10, 10], [-10, 10]],
           bins=(500, 500), cmin=0.1, cmap=plt.cm.get_cmap("hot"))
plt.colorbar()
plt.xlabel('x Vertex (cm)', fontsize=20)
plt.ylabel('y Vertex (cm)', fontsize=20)
plt.title('2D Vertex', fontsize=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(df.nBTOFmatch, bins=100, histtype='step')
plt.xlabel('BTOFMatch', fontsize=20)
plt.title('BTOFMatch', fontsize=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(df.VzVPD, range=[-500, 500], bins=1000, histtype='step')
plt.xlabel(r'$V_z$ (cm)', fontsize=20)
plt.title(r'$V_{z, VPD}$', fontsize=30)
plt.tight_layout()
plt.show()

# Now to make our event cuts.
dfCut = df[abs(df.VzVPD) <= 70]  # z vertex < 70 cm
dfCut = dfCut[dfCut.nBTOFmatch <= 100]
dfCut = dfCut[dfCut['rVert'] <= 2.0]  # r vertex < 2 cm
indexNamesArr = dfCut.index.values  # renaming the index
selector = np.linspace(0, len(df)-1, len(df), dtype=int)
selector = np.delete(selector, indexNamesArr)
rings = np.delete(rings, selector, axis=0)
indexNamesArr[:] = np.linspace(0, len(dfCut)-1, len(dfCut))[:]

plt.figure(figsize=(8, 5))
plt.hist2d(dfCut.xVert, dfCut.yVert, range=[[-10, 10], [-10, 10]],
           bins=(500, 500), cmin=0.1, cmap=plt.cm.get_cmap("jet"))
plt.colorbar()
plt.xlabel('x Vertex (cm)', fontsize=20)
plt.ylabel('y Vertex (cm)', fontsize=20)
plt.title('2D Vertex (after cuts)', fontsize=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(dfCut.nBTOFmatch, bins=100, histtype='step')
plt.xlabel('BTOFMatch', fontsize=20)
plt.title('BTOFMatch', fontsize=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(dfCut.VzVPD, range=[-500, 500], bins=1000, histtype='step')
plt.xlabel(r'$V_z$ (cm)', fontsize=20)
plt.title(r'$V_{z, VPD}$', fontsize=30)
plt.tight_layout()
plt.show()

dfCut.to_pickle('pandasdatacut.pkl')
np.save('ringscut.npy', rings)
