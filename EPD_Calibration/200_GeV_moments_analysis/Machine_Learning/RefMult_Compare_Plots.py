# \Display comparison between data RefMult3,
# EPD ring sums, LW, and ML predictions
#
#
# \author Skipper Kagamaster
# \date 04/29/2022
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import pandas as pd

file_loc = r'C:\200\PythonArrays\Analysis_Proton_Arrays'
os.chdir(file_loc)
df = pd.read_pickle('full_set.pkl')
refmult = df['RefMult3'].to_numpy()
ringsum = df['ring1'].to_numpy()
for i in (1, 31):
    ringsum = np.add(ringsum, df['ring{}'.format(i+1)].to_numpy())

# refmult = np.load(r'C:\200\ML\refmult.npy', allow_pickle=True)
# ringsum = np.load(r'C:\200\ML\ringsum.npy', allow_pickle=True)
predictions = np.load(r'C:\200\ML\predictions_200_refmult3.npy', allow_pickle=True)
LW = predictions[0]
ML = predictions[1]
EPD = [ringsum, LW, ML]

plt.hist(refmult, bins=1500, range=(0, 1500), histtype='step', density=True, color='k',
         alpha=0.6, label=r'$X_{RM3}$', lw=2)
# plt.hist(ringsum, bins=1500, range=(0, 1500), histtype='step', density=True, color='r',
#         alpha=0.6, label=r'$X_{\Sigma}$', lw=2)
plt.hist(LW, bins=1700, range=(-200, 1500), histtype='step', density=True, color='b',
         alpha=0.6, label=r'$X_{LW}$', lw=2)
plt.hist(ML, bins=1500,range=(0, 1500), histtype='step', density=True, color='orange',
         alpha=0.6, label=r'$X_{ReLU}$', lw=2)
plt.yscale('log')
plt.xlabel('X', fontsize=20, loc='right')
plt.ylabel(r'$\frac{dN}{dX}$', fontsize=20, loc='top')
plt.legend()
plt.show()

labels = [r'$X_{\Sigma}$', r'$X_{LW}$', r'$X_{ReLU}$']
fig, ax = plt.subplots(2, 2, figsize=(9, 9), constrained_layout=True)
ranges = ((0, 200), (-100, 1100), (0, 1300))
bins = (200, 1200, 1300)
for i in range(2):
    for j in range(2):
        x = i*2 + j
        if x > 2:
            ax[i, j].set_axis_off()
            continue
        count, binsX, binsY = np.histogram2d(refmult, EPD[x], bins=(1500, bins[x]),
                                             range=((0, 1500), ranges[x]))
        X, Y = np.meshgrid(binsY, binsX)
        im = ax[i, j].pcolormesh(Y, X, count, cmap='jet', norm=LogNorm())
        ax[i, j].set_ylabel(labels[x], fontsize=15)
        ax[i, j].set_xlabel(r'$X_{RM3}$', fontsize=15)
        fig.colorbar(im, ax=ax[i, j])
plt.show()
