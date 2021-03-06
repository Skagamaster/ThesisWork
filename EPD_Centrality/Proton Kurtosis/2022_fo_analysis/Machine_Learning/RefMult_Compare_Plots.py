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

refmult = np.load(r'D:\14GeV\Thesis\ML\refmult.npy', allow_pickle=True)
ringsum = np.load(r'D:\14GeV\Thesis\ML\ringsum.npy', allow_pickle=True)
predictions = np.load(r'D:\14GeV\Thesis\ML\predictions_14_refmult3.npy', allow_pickle=True)
LW = predictions[0]
ML = predictions[1]
EPD = [ringsum, LW, ML]

plt.hist(refmult, bins=700, range=(0, 700), histtype='step', density=True, color='k',
         alpha=0.6, label='RefMult3', lw=2)
plt.hist(ringsum, bins=300, histtype='step', density=True, color='r', alpha=0.6, label=r'$X_{\Sigma}$',
         lw=2)
plt.hist(LW, bins=300, histtype='step', density=True, color='b', alpha=0.6, label=r'$X_{LW}$',
         lw=2)
plt.hist(ML, bins=300, histtype='step', density=True, color='orange', alpha=0.6, label=r'$X_{ReLU}$',
         lw=2)
plt.yscale('log')
plt.xlabel('X', fontsize=20, loc='right')
plt.ylabel(r'$\frac{dN}{dX}$', fontsize=20, loc='top')
plt.legend()
plt.show()

labels = [r'$X_{\Sigma}$', r'$X_{LW}$', r'$X_{ReLU}$']
fig, ax = plt.subplots(2, 2, figsize=(9, 9), constrained_layout=True)
for i in range(2):
    for j in range(2):
        x = i*2 + j
        if x > 2:
            ax[i, j].set_axis_off()
            continue
        count, binsX, binsY = np.histogram2d(refmult, EPD[x], bins=700,
                                             range=((0, 700), (-100, 700)))
        X, Y = np.meshgrid(binsY, binsX)
        im = ax[i, j].pcolormesh(Y, X, count, cmap='jet', norm=LogNorm())
        ax[i, j].set_ylabel(labels[x], fontsize=15)
        ax[i, j].set_xlabel('RefMult3', fontsize=15)
        fig.colorbar(im, ax=ax[i, j])
plt.show()
