import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import uproot as up
import os
from matplotlib.colors import LogNorm

"""
from keras.models import load_model
from numba import vectorize
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, Conv1D
from keras.utils.generic_utils import get_custom_objects
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras.callbacks import EarlyStopping
import definitions as dfn
from definitions import Swish, swish, Mish, mish, Bose, bose
"""
print("Loading data.")
loc = r'C:\Users\dansk\Documents\Thesis\Protons\WIP\QA_4_14_2021\protons\total\ML_sets\\'
# Numpy array for all the ring values (1-16 East side, 17-32 West side, vzvpd)
# rings = np.load('{}rings.npy'.format(loc), allow_pickle=True)
# print("Ring data loaded.")
refmult3 = np.load('{}refmult3.npy'.format(loc), allow_pickle=True)
print("RefMult3 loaded.")
outer_rings = np.load("{}outer_ring_sums.npy".format(loc), allow_pickle=True)
print("Outer ring sums loaded.")
# rings = rings.astype('float32')
refmult3 = refmult3.astype('float32')
outer_rings = outer_rings.astype('float32')
print("Transformations done.")
MLP_predictions = np.load('{}relu_predictions.npy'.format(loc), allow_pickle=True)
print("RELU predictions loaded.")
linear_predictions = np.load('{}linear_predictions.npy'.format(loc), allow_pickle=True)
print("Linear predictions loaded.")
print("Data loaded. Now doing stuff.")

"""
# Run this if you have predictions to generate.
print("Generating outer ring sums.")
out1 = np.sum((rings.T[6:16]), axis=0)
out2 = np.sum((rings.T[22:32]), axis=0)
outer_rings = np.sum((out1, out2), axis=0)
print("Outer ring sums calculated. Saving the array.")
np.save("{}outer_ring_sums.npy".format(loc), outer_rings)
print("Outer ring sums saved.")
print("Basic data loaded. Importing models.")
# Load the models.
MLP = load_model('{}reluFitmodel_cut.h5'.format(loc))
linear = load_model('{}linearFitmodel_cut.h5'.format(loc))
print("Models imported. Generating predictions.")
MLP_predict = MLP.predict(rings)
print('RELU predictions generated.')
linear_predict = linear.predict(rings)
print('Linear predictions generated.')
print('Saving prediction arrays.')
np.save('relu_predictions.npy', MLP_predict)
print('RELU prediction array saved.')
np.save('linear_predictions.npy', linear_predict)
print('Linear prediction array saved.')
"""
# Plots of the centrality quantities.
plt.figure(figsize=(16, 9), constrained_layout=True)
alpha = 0.7
plt.hist(refmult3, bins=100, histtype='step', density=True, label="RefMult3", lw=2,
         alpha=alpha)
plt.hist(outer_rings, bins=100, histtype='step', density=True, label=r"$\Sigma EPD_{outer}$",
         alpha=alpha, lw=2)
plt.hist(linear_predictions, bins=100, histtype='step', density=True, label=r"$X_{W,\zeta'}$",
         alpha=alpha, lw=2)
plt.hist(MLP_predictions, bins=100, histtype='step', density=True, label="MLP NN", lw=2,
         alpha=alpha)
plt.yscale('log')
plt.ylabel(r"$\frac{dN}{dX}$", fontsize=15)
plt.xlabel("X", fontsize=15)
plt.title(r"Centrality Distributions at $\sqrt{s_{NN}}$= 14.5 GeV",
          fontsize=20)
plt.legend()
# plt.show()
plt.close()

RefCuts = [0, 10, 21, 41, 72, 118, 182, 270, 392, 472]  # Not in use right now.

# Let's use a simple percentile arrangement to get some centrality cuts.
quant_cuts = [95, 90, 80, 70, 60, 50, 40, 30, 20, 10]
length = int(len(quant_cuts))
MLP_bins = np.zeros(length)
linear_bins = np.zeros(length)
ref_bins = np.zeros(length)
outer_bins = np.zeros(length)
for i in range(length):
    MLP_bins[i] = np.percentile(MLP_predictions, quant_cuts[i])
    linear_bins[i] = np.percentile(linear_predictions, quant_cuts[i])
    ref_bins[i] = np.percentile(refmult3, quant_cuts[i])
    outer_bins[i] = np.percentile(outer_rings, quant_cuts[i])
# And now to get RefMult3 distributions from the percentile cuts.
ref_dists = []
ref_cums = [[], []]
outer_dists = []
outer_cums = [[], []]
linear_dists = []
linear_cums = [[], []]
MLP_dists = []
MLP_cums = [[], []]
ref_dists.append(refmult3[refmult3 >= ref_bins[0]])
ref_cums[0].append(np.mean(ref_dists[0]))
ref_cums[1].append(np.var(ref_dists[0]))
outer_dists.append(refmult3[outer_rings >= outer_bins[0]])
outer_cums[0].append(np.mean(outer_dists[0]))
outer_cums[1].append(np.var(outer_dists[0]))
linear_dists.append(refmult3[linear_predictions >= linear_bins[0]])
linear_cums[0].append(np.mean(linear_dists[0]))
linear_cums[1].append(np.var(linear_dists[0]))
MLP_dists.append(refmult3[MLP_predictions >= MLP_bins[0]])
MLP_cums[0].append(np.mean(MLP_dists[0]))
MLP_cums[1].append(np.var(MLP_dists[0]))
for i in range(length - 1):
    index_refmult3 = ((refmult3 < ref_bins[i]) & (refmult3 >= ref_bins[i + 1]))
    index_outer = ((outer_rings < outer_bins[i]) & (outer_rings >= outer_bins[i + 1]))
    index_linear = ((linear_predictions < linear_bins[i]) & (linear_predictions >= linear_bins[i + 1]))
    index_MLP = ((MLP_predictions < MLP_bins[i]) & (MLP_predictions >= MLP_bins[i + 1]))
    ref_dists.append(refmult3[index_refmult3])
    outer_dists.append(refmult3[index_outer])
    linear_dists.append(refmult3[index_linear])
    MLP_dists.append(refmult3[index_MLP])
    ref_cums[0].append(np.mean(ref_dists[i + 1]))
    ref_cums[1].append(np.var(ref_dists[i + 1]))
    outer_cums[0].append(np.mean(outer_dists[i + 1]))
    outer_cums[1].append(np.var(outer_dists[i + 1]))
    linear_cums[0].append(np.mean(linear_dists[i + 1]))
    linear_cums[1].append(np.var(linear_dists[i + 1]))
    MLP_cums[0].append(np.mean(MLP_dists[i + 1]))
    MLP_cums[1].append(np.var(MLP_dists[i + 1]))
ref_dists.append(refmult3[refmult3 < ref_bins[length - 1]])
ref_cums[0].append(np.mean(ref_dists[length - 1]))
ref_cums[1].append(np.var(ref_dists[length - 1]))
outer_dists.append(refmult3[outer_rings < outer_bins[length - 1]])
outer_cums[0].append(np.mean(outer_dists[length - 1]))
outer_cums[1].append(np.var(outer_dists[length - 1]))
linear_dists.append(refmult3[linear_predictions < linear_bins[length - 1]])
linear_cums[0].append(np.mean(linear_dists[length - 1]))
linear_cums[1].append(np.var(linear_dists[length - 1]))
MLP_dists.append(refmult3[MLP_predictions < MLP_bins[length - 1]])
MLP_cums[0].append(np.mean(MLP_dists[length - 1]))
MLP_cums[1].append(np.var(MLP_dists[length - 1]))
MLP_phi = np.divide(MLP_cums[1], ref_cums[1])
outer_phi = np.divide(outer_cums[1], ref_cums[1])
linear_phi = np.divide(linear_cums[1], ref_cums[1])

max_ref = np.max(refmult3)
min_ref = np.min(refmult3)
bins_ref = int(np.abs(max_ref - min_ref))
max_MLP = np.max(MLP_predictions)
min_MLP = np.min(MLP_predictions)
bins_MLP = int(np.abs(max_MLP - min_MLP))
max_linear = np.max(linear_predictions)
min_linear = np.min(linear_predictions)
bins_linear = int(np.abs(max_linear - min_linear))
max_outer = np.max(outer_rings)
min_outer = np.min(outer_rings)
bins_outer = int(np.abs(max_outer - min_outer))
print("Plotting correlations.")
fig, ax = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)
bins = 600
ranger = (0, 600)
legend_names = ["RefMult3", r"$Ref_{\Sigma EPD_{outer}}$", r"$Ref_{X_{W,\zeta'}}$", r"$Ref_{ML-NN}$",
                '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_',
                '_nolegend_', '_nolegend_', '_nolegend_']
count = 0
for i in (0, 3, 9):
    ax[0].hist(ref_dists[i], bins=bins, histtype='step', label=legend_names[count],
               color="black", range=ranger, density=True)
    count += 1
    ax[0].hist(outer_dists[i], bins=bins, histtype='step', label=legend_names[count],
               color="blue", range=ranger, density=True)
    count += 1
    ax[0].hist(linear_dists[i], bins=bins, histtype='step', label=legend_names[count],
               color="red", range=ranger, density=True)
    count += 1
    ax[0].hist(MLP_dists[i], bins=bins, histtype='step', label=legend_names[count],
               color="green", range=ranger, density=True)
    count += 1
ax[0].legend()
ax[0].set_xlabel("X", fontsize=15)
ax[0].set_ylabel(r"$\frac{dP}{dX}$", fontsize=15)
ax[0].set_yscale('log')
ax[0].grid()
ax[0].set_ylim([5e-5, 0.5])

x = np.linspace(0, 10, 11).astype(int)
x_labels = ["90-100%", "80-90%", "70-80%", "60-70%", "50-60%", "40-50%", "30-40%", "20-30%", "10-20%", "5-10%", "0-5%"]
ax[1].plot(outer_phi[::-1], lw=2, ls=":", marker="o", ms=10, c="blue", label=r"$Ref_{\Sigma EPD_{outer}}$")
ax[1].plot(linear_phi[::-1], lw=2, ls=":", marker="^", ms=10, c="red", label=r"$Ref_{X_{W,\zeta'}}$")
ax[1].plot(MLP_phi[::-1], lw=2, ls=":", marker="*", ms=10, c="green", label=r"$Ref_{ML-NN}$")
ax[1].set_yscale('log')
ax[1].set_xticks(x)
ax[1].set_xticklabels(x_labels)
ax[1].legend()
ax[1].grid()
ax[1].set_ylabel(r"$\Phi = \frac{\sigma_X}{\sigma_{refmult3}}$", fontsize=15)
ax[1].set_xlabel("% Centrality")

plt.show()
plt.close()

fig, ax = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
counts, binsX, binsY = np.histogram2d(outer_rings, refmult3, bins=(bins_outer, bins_ref),
                                      range=((min_outer, max_outer), (min_ref, max_ref)))
im_0 = ax[0].pcolormesh(binsY[:-1], binsX[:-1], counts, cmap="jet", norm=colors.LogNorm(),
                        shading='auto')
ax[0].set_xlabel("RefMult3", fontsize=10)
ax[0].set_ylabel(r"$\Sigma EPD_{outer}$", fontsize=10)
ax[0].set_title(r"RefMult3 vs $\Sigma EPD_{outer}$", fontsize=20)
fig.colorbar(im_0, ax=ax[0])

counts, binsX, binsY = np.histogram2d(linear_predictions, refmult3, bins=(bins_linear, bins_ref),
                                      range=((min_linear, max_linear), (min_ref, max_ref)))
im_1 = ax[1].pcolormesh(binsY[:-1], binsX[:-1], counts, cmap="jet", norm=colors.LogNorm(),
                        shading='auto')
ax[1].set_xlabel("RefMult3", fontsize=10)
ax[1].set_ylabel(r"$X_{W,\zeta'}$", fontsize=10)
ax[1].set_title(r"RefMult3 vs $X_{W,\zeta'}$", fontsize=20)
fig.colorbar(im_1, ax=ax[1])

counts, binsX, binsY = np.histogram2d(MLP_predictions, refmult3, bins=(bins_MLP, bins_ref),
                                      range=((min_MLP, max_MLP), (min_ref, max_ref)))
im_2 = ax[2].pcolormesh(binsY[:-1], binsX[:-1], counts, cmap="jet", norm=colors.LogNorm(),
                        shading='auto')
ax[2].set_xlabel("RefMult3", fontsize=10)
ax[2].set_ylabel(r"MLP NN", fontsize=10)
ax[2].set_title(r"RefMult3 vs MLP NN", fontsize=20)
fig.colorbar(im_2, ax=ax[2])

plt.show()
plt.close()
