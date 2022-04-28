# \author Skipper Kagamaster
# \date 06/24/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
"""
The purpose of this code is to compare cumulants (no CBWC yet) with different
determinations of centrality.
"""

import uproot as up
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import skew, kurtosis
import os
import json

single = False

# Load predictions and refmult.
if single is True:
    refmult = np.loadtxt(r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results\BigList\out_1.txt")[:, 1]
    net_protons = np.loadtxt(r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results\BigList\out_1.txt")[:, 4]
    linear = np.load(r"C:\Users\dansk\Documents\Thesis\2021_DNP_Work\ML_fits\linearpredictions_single.npy",
                     allow_pickle=True)
    relu = np.load(r"C:\Users\dansk\Documents\Thesis\2021_DNP_Work\ML_fits\relupredictions_single.npy",
                   allow_pickle=True)
    swish = np.load(r"C:\Users\dansk\Documents\Thesis\2021_DNP_Work\ML_fits\swishpredictions_single.npy",
                    allow_pickle=True)
else:
    cutoff = int(-1)
    proton_params = pd.read_pickle(
        r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results\BigList\out_all.pkl")[:cutoff]
    net_protons = proton_params['net_protons'].to_numpy()
    refmult = proton_params['refmult'].to_numpy()
    linear = np.load(r"C:\Users\dansk\Documents\Thesis\2021_DNP_Work\ML_fits\linearpredictions.npy",
                     allow_pickle=True)[:cutoff]
    relu = np.load(r"C:\Users\dansk\Documents\Thesis\2021_DNP_Work\ML_fits\relupredictions.npy",
                   allow_pickle=True)[:cutoff]
    swish = np.load(r"C:\Users\dansk\Documents\Thesis\2021_DNP_Work\ML_fits\swishpredictions.npy",
                    allow_pickle=True)[:cutoff]
# Make predictions integers (for comparison's sake, at least).
refmult = refmult.astype('int')
linear = linear.astype('int')
relu = relu.astype('int')
swish = swish.astype('int')
swish[swish == 0] = 0
# Combine the centrality observables in a list.
centralities = [refmult, linear, relu, swish]
types = ["refmult", "linear", "relu", "swish"]
types = ["RefMult3", r"$\xi_{Linear}$", r"$\xi_{ReLU}$", r"$\xi_{Swish}$"]

# This will build the centrality cuts.
RefCuts = [-999, 10, 21, 41, 72, 118, 182, 270, 392, 472]  # Yu's determination of RefMult3 centrality via Glauber.
ref_len = len(refmult)
lengths = []
for i in range(len(RefCuts)-1):
    lengths.append(len(refmult[refmult < RefCuts[i+1]])/ref_len)
quants = []
color = ["blue", "orange", "green", "red"]
for i in range(len(centralities)):
    quants.append(np.quantile(centralities[i], lengths))
    count, bins = np.histogram(centralities[i], bins=850, range=(-150, 700), density=True)
    plt.plot(bins[:-1], count, label=types[i], lw=3, alpha=0.7, color=color[i])
    # plt.hist(centralities[i], bins=850, range=(-150, 700), histtype='step',
    #         density=True, label=types[i], lw=3, alpha=0.7)
    plt.axvline(x=quants[i][-1], color=color[i], alpha=0.7)
    print(int(quants[i][-1]))
quants = np.array(quants)
plt.legend()
plt.title("Centrality Distributions", fontsize=30)
plt.xlabel("C (centrality #)", fontsize=20)
plt.xlim((-150, 700))
plt.ylabel(r"$\frac{dN}{dC}$", fontsize=20)
plt.yscale('log')
plt.show()
plt.close()

# Now to get the cumulants from our data. Also, I'll get an n array for CBWC later.
protons = []
x_vals = []
n_arr = []
for j in range(len(centralities)):
    n_arr.append([])
    unique = np.unique(centralities[j])
    x_vals.append(unique)
    size = len(unique)
    pro_cumulants = np.zeros((4, size))
    for i in range(size):
        pro_int = net_protons[centralities[j] == unique[i]]
        n_arr[j].append(len(pro_int))
        pro_cumulants[0][i] = np.mean(pro_int)
        pro_cumulants[1][i] = np.var(pro_int)
        pro_cumulants[2][i] = skew(pro_int) * np.power(np.sqrt(np.var(pro_int)), 3)
        pro_cumulants[3][i] = kurtosis(pro_int) * np.power(np.var(pro_int), 2)
    protons.append(pro_cumulants)
protons = ak.Array(protons)
n_arr = ak.Array(n_arr)
x_vals = ak.Array(x_vals)
with open(r'C:\Users\dansk\Documents\Thesis\2021_DNP_Work\arrays\protons.txt', 'w') as f:
    json.dump(ak.to_list(protons), f)
with open(r'C:\Users\dansk\Documents\Thesis\2021_DNP_Work\arrays\n_arr.txt', 'w') as f:
    json.dump(ak.to_list(n_arr), f)
with open(r'C:\Users\dansk\Documents\Thesis\2021_DNP_Work\arrays\x_vals.txt', 'w') as f:
    json.dump(ak.to_list(x_vals), f)
np.save(r'C:\Users\dansk\Documents\Thesis\2021_DNP_Work\arrays\quants.npy', quants)

# Now let's see how we did.
fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
titles = [r"$C_1$", r"$C_2$", r"$C_3$", r"$C_4$"]
for k in range(len(centralities)):
    for i in range(2):
        for j in range(2):
            x = 2*i + j
            ax[i, j].scatter(x_vals[k], protons[k][x], label=types[k], alpha=0.5)
            ax[i, j].set_xlabel("Centrality_Value", fontsize=15)
            ax[i, j].set_ylabel(titles[x], fontsize=20)
ax[1, 1].legend()
plt.suptitle("Cumulants Comparison", fontsize=30)
plt.show()
plt.close()

# Now to do the same but in RefMult3 determined bins (no CBWC).
protons_bins = []
for j in range(len(centralities)):
    pro_cumulants = np.zeros((4, len(quants[j])))
    for i in range(len(quants[j])-1):
        pro_int = net_protons[(centralities[j] >= quants[j][i]) & (centralities[j] < quants[j][i+1])]
        pro_cumulants[0][i] = np.mean(pro_int)
        pro_cumulants[1][i] = np.var(pro_int)
        pro_cumulants[2][i] = skew(pro_int) * np.power(np.sqrt(np.var(pro_int)), 3)
        pro_cumulants[3][i] = kurtosis(pro_int) * np.power(np.var(pro_int), 2)
    final_pro = net_protons[centralities[j] >= quants[j][-1]]
    pro_cumulants[0][len(quants[j])-1] = np.mean(final_pro)
    pro_cumulants[1][len(quants[j])-1] = np.var(final_pro)
    pro_cumulants[2][len(quants[j])-1] = skew(final_pro) * np.power(np.sqrt(np.var(final_pro)), 3)
    pro_cumulants[3][len(quants[j])-1] = kurtosis(final_pro) * np.power(np.var(final_pro), 2)
    protons_bins.append(pro_cumulants)

# Let's again see how we did.
types = ["refmult", "linear", "relu", "swish"]
fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
marker = ["o", "*", "^", "P"]
for k in range(len(centralities)):
    for i in range(2):
        for j in range(2):
            x = 2*i + j
            ax[i, j].plot(protons_bins[k][x], label=types[k], alpha=0.5,
                          lw=0, marker=marker[k], ms=10)
            ax[i, j].set_xlabel("Centrality_Value", fontsize=15)
            ax[i, j].set_ylabel(titles[x], fontsize=20)
ax[1, 1].legend()
plt.suptitle("Cumulants Comparison", fontsize=30)
plt.show()
plt.close()

# Now let's add the CBWC.
protons_cbwc_int = []
protons_cbwc = []
for i in range(len(centralities)):
    protons_cbwc_int.append([])
    for k in range(len(protons[i])):
        protons_cbwc_int[i].append(protons[i][k]*n_arr[i])
    pro_cumulants_cbwc = np.zeros((4, len(quants[i])))
    for j in range(len(quants[i]) - 1):
        for l in range(4):
            cbwc_int = protons_cbwc_int[i][l][(centralities[i] >= quants[i][j]) & (centralities[i] < quants[i][j + 1])]
            pro_cumulants_cbwc[l][i] = np.divide(np.sum(cbwc_int[l]), np.sum(n_arr[i][l]))
    for m in range(4):
        final_pro_cbwc = protons_cbwc_int[i][m][centralities[i] >= quants[i][-1]]
        pro_cumulants_cbwc[m][len(quants[i]) - 1] = np.divide(np.sum(final_pro_cbwc), np.sum(n_arr[i][m]))
    protons_cbwc.append(pro_cumulants_cbwc)

# Let's again see how we did.
types = ["refmult", "linear", "relu", "swish"]
fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
for k in range(len(centralities)):
    for i in range(2):
        for j in range(2):
            x = 2*i + j
            ax[i, j].plot(protons_cbwc[k][x], label=types[k], alpha=0.5,
                          lw=0, marker=marker[k], ms=10)
            ax[i, j].set_xlabel("Centrality_Value", fontsize=15)
            ax[i, j].set_ylabel(titles[x], fontsize=20)
ax[1, 1].legend()
plt.suptitle("Cumulants Comparison", fontsize=30)
plt.show()
plt.close()
