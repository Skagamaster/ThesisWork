import numpy as np
from scipy.stats import skew, kurtosis
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import uproot as up
import os
from matplotlib.colors import LogNorm

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
protons = np.load("{}protons.npy".format(loc), allow_pickle=True)
antiprotons = np.load("{}antiprotons.npy".format(loc), allow_pickle=True)
net_protons = np.subtract(protons, antiprotons)
print("Protons all loaded.")
print("Data loaded. Now doing stuff.")

print("Converting to integers.")
MLP_predictions = MLP_predictions.astype(int)
linear_predictions = linear_predictions.astype(int)
outer_rings = outer_rings.astype(int)

# Let's use a simple percentile arrangement to get some centrality cuts.
print("Calculating quantiles.")
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
print("RefMult3:", ref_bins)
print("Outer sum:", outer_bins)
print("Linear weight:", linear_bins)
print("MLP:", MLP_bins)
print("Quantile calculations complete.")

# Let's make our cumulant arrays. This might take a bit ...
print("Now for the cumulants.")
# Cumulant analysis.
cumulant_names = np.asarray(("Mean", "Variance", "Skewness", "Kurtosis"))
# Net proton distributions from the percentile cuts.
print("Generating proton distributions.")
ref_dists = []
outer_dists = []
linear_dists = []
MLP_dists = []
ref_dists.append(net_protons[refmult3 >= ref_bins[0]])
outer_dists.append(net_protons[outer_rings >= outer_bins[0]])
linear_dists.append(net_protons[linear_predictions >= linear_bins[0]])
MLP_dists.append(net_protons[MLP_predictions >= MLP_bins[0]])
for i in range(length - 1):
    index_refmult3 = ((refmult3 < ref_bins[i]) & (refmult3 >= ref_bins[i + 1]))
    index_outer = ((outer_rings < outer_bins[i]) & (outer_rings >= outer_bins[i + 1]))
    index_linear = ((linear_predictions < linear_bins[i]) & (linear_predictions >= linear_bins[i + 1]))
    index_MLP = ((MLP_predictions < MLP_bins[i]) & (MLP_predictions >= MLP_bins[i + 1]))
    ref_dists.append(net_protons[index_refmult3])
    outer_dists.append(net_protons[index_outer])
    linear_dists.append(net_protons[index_linear])
    MLP_dists.append(net_protons[index_MLP])
ref_dists.append(net_protons[refmult3 < ref_bins[length - 1]])
outer_dists.append(net_protons[outer_rings < outer_bins[length - 1]])
linear_dists.append(net_protons[linear_predictions < linear_bins[length - 1]])
MLP_dists.append(net_protons[MLP_predictions < MLP_bins[length - 1]])
print("Distributions complete.")

print("Calculating cumulants (no CBWC).")
# Cumulants for centrality ranges, but with no CBWC.
cumulants_ref = np.zeros((4, length+1))
cumulants_out = np.zeros((4, length+1))
cumulants_lin = np.zeros((4, length+1))
cumulants_MLP = np.zeros((4, length+1))
for i in range(length+1):
    print("Working on", i, "of", length)
    cumulants_ref[0][i] = np.mean(ref_dists[i])
    cumulants_ref[1][i] = np.var(ref_dists[i])
    cumulants_ref[2][i] = skew(ref_dists[i]) * np.power(np.sqrt(np.var(ref_dists[i])), 3)
    cumulants_ref[3][i] = kurtosis(ref_dists[i]) * np.power(np.var(ref_dists[i]), 2)
    cumulants_out[0][i] = np.mean(outer_dists[i])
    cumulants_out[1][i] = np.var(outer_dists[i])
    cumulants_out[2][i] = skew(outer_dists[i]) * np.power(np.sqrt(np.var(outer_dists[i])), 3)
    cumulants_out[3][i] = kurtosis(outer_dists[i]) * np.power(np.var(outer_dists[i]), 2)
    cumulants_lin[0][i] = np.mean(linear_dists[i])
    cumulants_lin[1][i] = np.var(linear_dists[i])
    cumulants_lin[2][i] = skew(linear_dists[i]) * np.power(np.sqrt(np.var(linear_dists[i])), 3)
    cumulants_lin[3][i] = kurtosis(linear_dists[i]) * np.power(np.var(linear_dists[i]), 2)
    cumulants_MLP[0][i] = np.mean(MLP_dists[i])
    cumulants_MLP[1][i] = np.var(MLP_dists[i])
    cumulants_MLP[2][i] = skew(MLP_dists[i]) * np.power(np.sqrt(np.var(MLP_dists[i])), 3)
    cumulants_MLP[3][i] = kurtosis(MLP_dists[i]) * np.power(np.var(MLP_dists[i]), 2)
print("Cumulants calculated (no CBWC).")
RefCuts_labels = ["0-5%", "5-10%", "10-20%", "20-30%", "30-40%", "40-50%",
                  "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
x_tic = np.linspace(0, length, length+1)
fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
# Get the cumulants in the "normal" form.
cumulants_ref[1] = np.divide(cumulants_ref[1], cumulants_ref[0])
cumulants_ref[2] = np.divide(cumulants_ref[2], cumulants_ref[1])
cumulants_ref[3] = np.divide(cumulants_ref[3], cumulants_ref[1])

cumulants_out[1] = np.divide(cumulants_out[1], cumulants_out[0])
cumulants_out[2] = np.divide(cumulants_out[2], cumulants_out[1])
cumulants_out[3] = np.divide(cumulants_out[3], cumulants_out[1])

cumulants_lin[1] = np.divide(cumulants_lin[1], cumulants_lin[0])
cumulants_lin[2] = np.divide(cumulants_lin[2], cumulants_lin[1])
cumulants_lin[3] = np.divide(cumulants_lin[3], cumulants_lin[1])

cumulants_MLP[1] = np.divide(cumulants_MLP[1], cumulants_MLP[0])
cumulants_MLP[2] = np.divide(cumulants_MLP[2], cumulants_MLP[1])
cumulants_MLP[3] = np.divide(cumulants_MLP[3], cumulants_MLP[1])
for i in range(2):
    for j in range(2):
        x = i * 2 + j
        ax[i, j].plot(x_tic, cumulants_ref[x][::-1], c='black', marker="o", mfc=None, lw=0, mew=2,
                      label="refmult3", alpha=0.5, ms=10)
        ax[i, j].plot(x_tic, cumulants_out[x][::-1], c='blue', marker="^", mfc=None, lw=0, mew=2,
                      label="outer", alpha=0.5, ms=10)
        ax[i, j].plot(x_tic, cumulants_lin[x][::-1], c='red', marker="*", mfc=None, lw=0, mew=2,
                      label="linear", alpha=0.5, ms=10)
        ax[i, j].plot(x_tic, cumulants_MLP[x][::-1], c='green', marker="v", mfc=None, lw=0, mew=2,
                      label="MLP", alpha=0.5, ms=10)
        ax[i, j].set_title(cumulant_names[x])
        ax[i, j].set_xticks(x_tic)
        ax[i, j].set_xticklabels(RefCuts_labels[::-1], rotation=45)
        ax[i, j].set_xlabel("Centrality")
        ax[i, j].set_yscale('log')
        ax[i, j].grid(True)
        ax[i, j].legend()
fig.suptitle(r"Net Proton Cumulants $\sqrt{s_{NN}}$= 14.5 GeV", fontsize=20)
# plt.show()
plt.close()

# Just to compare with Yu's results.
C1 = np.array([0.5, 0.95, 1.58, 2.88, 4.61, 7.09, 10.59, 14.0, 16.78])
C2 = np.array([0.17, 0.78, 1.65, 2.99, 4.85, 7.53, 11.26, 14.89, 17.79])
C3 = np.array([0.17, 0.63, 1.30, 2.39, 3.87, 6.10, 9.10, 11.96, 13.79])
C4 = np.array([0.51, 1.10, 1.78, 2.54, 4.24, 6.53, 9.75, 13.05, 15.34])

yu = np.array((C1, np.divide(C2, C1), np.divide(C3, C2), np.divide(C4, C2)))
fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)

for i in range(2):
    for j in range(2):
        x = i * 2 + j
        ax[i, j].plot(x_tic, cumulants_ref[x][::-1], c='black', marker="o", mfc=None, lw=0, mew=2,
                      label="refmult3", alpha=0.5, ms=10)
        ax[i, j].plot(x_tic[2:], yu[x], c='blue', marker="^", mfc=None, lw=0, mew=2,
                      label="yu_values", alpha=0.5, ms=10)
        ax[i, j].set_title(cumulant_names[x])
        ax[i, j].set_xticks(x_tic)
        ax[i, j].set_xticklabels(RefCuts_labels[::-1], rotation=45)
        ax[i, j].set_xlabel("Centrality")
        ax[i, j].set_yscale('log')
        ax[i, j].grid(True)
        ax[i, j].legend()
fig.suptitle(r"Net Proton Cumulants $\sqrt{s_{NN}}$= 14.5 GeV", fontsize=20)
plt.show()
plt.close()

# And with CBWC.
proton_len = np.zeros(len(proton_dists))
c_for_cbwc = np.zeros((4, len(proton_dists)))
for i in range(len(proton_dists)):
    flat_dist = np.hstack(proton_dists[i])
    if len(flat_dist) == 0:
        continue
    else:
        proton_len[i] = len(flat_dist)
        c_for_cbwc[0][i] = np.mean(flat_dist)*len(flat_dist)
        c_for_cbwc[1][i] = np.var(flat_dist)*len(flat_dist)
        c_for_cbwc[2][i] = proton_len[i] * skew(flat_dist) * np.power(np.sqrt(np.var(flat_dist)), 3)
        c_for_cbwc[3][i] = kurtosis(flat_dist) * np.power(np.var(flat_dist), 2)*len(flat_dist)
cumulants_cbwc = [[], [], [], []]
for i in range(len(Ref3Cuts)-1):
    for j in range(4):
        cumulants_cbwc[j].append(np.sum(c_for_cbwc[j][Ref3Cuts[i]:Ref3Cuts[i+1]])
                                 / np.sum(proton_len[Ref3Cuts[i]:Ref3Cuts[i+1]]))
for j in range(4):
    cumulants_cbwc[j].append(np.sum(c_for_cbwc[j][Ref3Cuts[len(Ref3Cuts)-1]:])
                             / np.sum(proton_len[Ref3Cuts[len(Ref3Cuts)-1]:]))


fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
for i in range(2):
    for j in range(2):
        x = 2*i + j
        ax[i, j].plot(cumulants_ref[x][::-1])
        ax[i, j].plot(cumulants_out[x][::-1])
        ax[i, j].plot(cumulants_lin[x][::-1])
        ax[i, j].plot(cumulants_MLP[x][::-1])
plt.show()
