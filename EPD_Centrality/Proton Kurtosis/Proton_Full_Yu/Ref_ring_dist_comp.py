import numpy as np
import pandas as pd
import math
import uproot as up
import os
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

df = pd.read_pickle(r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results\BigList\out_all.pkl")
ring_set = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
ring_set = np.linspace(1, 32, 32).astype('int')
ring_sum = np.zeros(len(df))
rings = []
# Sum up all the outer ring sums (both sides of the EPD).
for i in ring_set:
    ring_sum = ring_sum + df["ring{}".format(i)].to_numpy()
    rings.append(df["ring{}".format(i)].to_numpy())
ring_sum[ring_sum >= 353] = 353
ring_sum = np.round(ring_sum).astype('int')
rings = np.array(rings)
ring_14 = df["ring14"].to_numpy()
ring_30 = df["ring30"].to_numpy()
refmult = df["refmult"].to_numpy()

plt.hist(rings[0], bins=200, histtype="step", density=True)
plt.xlabel(r"$\Sigma_{ring1}$", fontsize=15)
plt.ylabel(r"$\frac{dN}{d\Sigma_{ring1}}$", fontsize=15)
plt.title(r"$\Sigma$ for EPD Ring 1", fontsize=25)
plt.yscale("log")
plt.show()

counter, binsY, binsX = np.histogram2d(rings[0], refmult, bins=(100, 700), range=((0, 100), (0, 700)))
X, Y = np.meshgrid(binsX[:-1], binsY[:-1])
plt.pcolormesh(X, Y, counter, norm=colors.LogNorm(), cmap="jet", shading="auto")
plt.xlabel("RefMult3", fontsize=15)
plt.ylabel("EPD RingSum (1)", fontsize=15)
plt.title("RefMult3 vs EPD Ring 1", fontsize=25)
plt.colorbar()
plt.show()

fig, ax = plt.subplots(4, 4, figsize=(16, 9), constrained_layout=True)
for i in range(4):
    for j in range(4):
        x = i*4 + j
        # if x > 9:
        #    ax[i, j].axis("off")
        #    continue
        counter, binsY, binsX = np.histogram2d(rings[x], refmult, bins=(80, 700), range=((0, 80), (0, 700)))
        X, Y = np.meshgrid(binsX[:-1], binsY[:-1])
        im = ax[i, j].pcolormesh(X, Y, counter, norm=colors.LogNorm(), cmap="jet", shading="auto")
        ax[i, j].set_title("Ring {}".format(x+1))
        fig.colorbar(im, ax=ax[i, j])
plt.show()

fig, ax = plt.subplots(4, 4, figsize=(16, 9), constrained_layout=True)
for i in range(4):
    for j in range(4):
        x = i*4 + j
        # if x > 9:
        #    ax[i, j].axis("off")
        #    continue
        counter, binsY, binsX = np.histogram2d(rings[x+16], refmult, bins=(160, 700), range=((0, 80), (0, 700)))
        X, Y = np.meshgrid(binsX[:-1], binsY[:-1])
        im = ax[i, j].pcolormesh(X, Y, counter, norm=colors.LogNorm(), cmap="jet", shading="auto")
        ax[i, j].set_title("Ring {}".format(x + 17))
        fig.colorbar(im, ax=ax[i, j])
plt.show()
