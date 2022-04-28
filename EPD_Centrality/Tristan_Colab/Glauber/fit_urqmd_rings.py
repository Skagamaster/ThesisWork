import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
from scipy.optimize import curve_fit
import fit_rings_functions as frf
import time
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
from numpy.random import default_rng

rng = default_rng()

# Load the outer ring sum and n_coll/n_part data.
ring_sum, n_coll, n_part, predictions, refmult = frf.load_data()
# Just a thing to test out some known GMC parameters.
xlabels = [r'$X_{RM3}$', r'$X_{RM1}$', r'$X_{LW}$',
           r'$X_{ReLU}$', r'$X_{swish}$', r'$X_{CNN}$']
save_labels = ['X_RM3', 'X_RM1', 'b', 'X_LW',
               'X_ReLU', 'X_swish', 'X_CNN']
bin_max = 2500

# Now to get the RefMult3 values from UrQMD
ref_count = np.load(r'C:\Users\dansk\Documents\Thesis\Protons\2022_urqmd\pro_count_archives\1M_events_refmult3_0.4_2'
                    r'.0\pro_counts.npy', allow_pickle=True).astype(float)
ref_bin = np.load(r'C:\Users\dansk\Documents\Thesis\Protons\2022_urqmd\pro_count_archives\1M_events_refmult3_0.4_2'
                  r'.0\pro_bins.npy', allow_pickle=True)
refmult3 = np.repeat(ref_bin[2][0][0][:-1], np.sum(ref_count[2][0], axis=1).astype('int'))
refmult1 = np.repeat(ref_bin[2][1][0][:-1], np.sum(ref_count[2][1], axis=1).astype('int'))
b = np.repeat(ref_bin[2][2][0][:-1], np.sum(ref_count[2][2], axis=1).astype('int'))
X_lw = np.repeat(ref_bin[2][3][0][:-1], np.sum(ref_count[2][3], axis=1).astype('int'))
X_relu = np.repeat(ref_bin[2][4][0][:-1], np.sum(ref_count[2][4], axis=1).astype('int'))
X_swish = np.repeat(ref_bin[2][5][0][:-1], np.sum(ref_count[2][5], axis=1).astype('int'))
X_cnn = np.repeat(ref_bin[2][6][0][:-1], np.sum(ref_count[2][6], axis=1).astype('int'))

refs = [refmult3, refmult1, X_lw, X_relu, X_swish, X_cnn]
refs_ranges = [[100, 699], [100, 400], [100, 655], [100, 626], [100, 614], [100, 653]]
refs_ranges = [[100, bin_max], [100, bin_max], [100, bin_max], [100, bin_max], [100, bin_max], [100, bin_max]]

# Now to make the GMC distributions and fit the histograms.
dim = 20
n = np.linspace(1.0, 5.0, dim)
p = np.linspace(0.5, 0.8, dim)
alpha = np.linspace(0.3, 0.8, dim)

gmc_histos = []
optimals = np.load(r'C:\Users\dansk\Documents\Thesis\Protons\2022_urqmd\mse_mins.npy')

with PdfPages(r'C:\Users\dansk\Documents\Thesis\Protons\2022_urqmd\Images\gmc_histos.pdf') as pdf:
    fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for i in range(6):
        x = int(i / 3)
        j = i % 3
        gmc_histos.append([])
        apart = optimals[i][0][0] * n_part
        acoll = (1 - optimals[i][0][0]) * n_coll
        g_arr = np.add(apart, acoll) * optimals[i][0][1]
        g_arr[g_arr == 0] = 1e-6
        gmc = rng.negative_binomial(g_arr, optimals[i][0][2])
        count, bins = np.histogram(gmc, bins=750, range=(0, 750))
        count = count / count[100]
        refref, binnn = np.histogram(refs[i], bins=750, range=(0, 750))
        refref = refref / refref[100]
        count = np.hstack((0, count))
        refref = np.hstack((0, refref))
        ax[x, j].plot(bins, count, color='blue', label='GMC')
        ax[x, j].plot(bins, refref, color='red', label=xlabels[i], alpha=0.5)
        ax[x, j].set_xlabel(r"$X_{cent}$", fontsize=20)
        ax[x, j].set_ylabel(r"N/N[100]", fontsize=20)
        ax[x, j].legend()
        ax[x, j].set_yscale('log')
    pdf.savefig()
    plt.show()
    plt.close()

a_part = np.outer(alpha, n_part)
a_coll = np.outer(1 - alpha, n_coll)
garr = np.add(a_part, a_coll)
garr = np.reshape(np.outer(n, garr), (dim, dim, len(n_coll)))
garr[garr == 0] = 1e-6

ref_hists = []
for i in range(6):
    refref, binnn = np.histogram(refs[i], bins=bin_max, range=(0, bin_max))
    refref = np.where(refref > 0, refref / refref[100], 0)
    ref_hists.append(refref)
print("Begin your attack run!")
ref3_mse = [[], [], [], [], [], []]
time0 = time.time()
counter = 0
for i in p:
    print(counter + 1, "of", len(p))
    print("Processing NBD ...")
    rngtime = time.time()
    arr = rng.negative_binomial(garr.T, i).T
    for m in range(6):
        ref3_mse[m].append([])
    counter_ = 0
    print("NBD complete in:", round(time.time() - rngtime, 2),
          "s")
    print("Processing MSE ...")
    timej = time.time()
    for j in arr:
        for m in range(6):
            ref3_mse[m][counter].append([])
        for k in j:
            count, bin_ = np.histogram(k, bins=bin_max, range=(0, bin_max))
            count = np.where(count > 0, count / count[100], 0)
            """
            count_end = np.where(count[50:] == 0)
            if len(count_end[0]) == 0:
                for m in range(6):
                    ref3_mse[m][counter][counter_].append(1)
                continue
            count_end = count_end[0][0]
            """
            for m in range(6):
                """
                ref_end = np.where(ref_hists[m][50:] == 0)[0][0]
                if np.abs(bin_[50:][ref_end] - bin_[50:][count_end]) > 150:
                    # print("Too large a gap:", np.abs(bin_[50:][ref_end] - bin_[50:][count_end]))
                    ref3_mse[m][counter][counter_].append(1)
                    continue
                """
                mse = np.sum(np.subtract(ref_hists[m][refs_ranges[m][0]:refs_ranges[m][1]],
                                         count[refs_ranges[m][0]:refs_ranges[m][1]])) / \
                      len(count[refs_ranges[m][0]:refs_ranges[m][1]])
                ref3_mse[m][counter][counter_].append(mse)
        counter_ += 1
    print("MSE complete in:", round(time.time() - timej, 2), 's')
    counter += 1
print("Total time:", round((time.time() - time0) / 60, 2), 'minutes')
ref3_mse = np.asarray(ref3_mse)
np.save(r'C:\Users\dansk\Documents\Thesis\Protons\2022_urqmd\ref3_mse.npy', ref3_mse)
print("MSE saved.")
ref3_mse[np.isnan(ref3_mse)] = 500
mins = []
mse_mins = []
for i in range(6):
    mse_mins.append([])
    min_val = np.min(ref3_mse[i])
    mins.append(min_val)
    a_val = np.where(ref3_mse[i] == min_val)[0][0]
    n_val = np.where(ref3_mse[i] == min_val)[1][0]
    p_val = np.where(ref3_mse[i] == min_val)[2][0]
    mse_mins[i].append(np.asarray((alpha[a_val], n[n_val], p[p_val])))
mins = np.asarray(mins)
mse_mins = np.asarray(mse_mins)
np.save(r'C:\Users\dansk\Documents\Thesis\Protons\2022_urqmd\mse_mins.npy', mse_mins)
np.save(r'C:\Users\dansk\Documents\Thesis\Protons\2022_urqmd\mins.npy', mins)
