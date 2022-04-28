# \author Skipper Kagamaster
# \date 08/26/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
"""
The purpose of this code is to find outlying proton distributions by run
and truncate them so as not to pollute data with low proton runs.
Basically this is proton QA.
"""

import functions as fn
import uproot as up
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cbook
from scipy.stats import skew, kurtosis
import os
from pathlib import Path

# Use my files, or Yu's from RACF?
use_my_files = False
use_yu_files = False
use_ml_sets = True
use_my_pandas = False

# Yu's cuts for RefMult3:
ref_cuts = np.asarray((0, 10, 21, 41, 72, 118, 182, 270, 392, 472))
ref_cuts = np.asarray([0, 18, 34, 58, 92, 138, 197, 278, 388, 462])
rsum_cuts = np.asarray((0, 53, 73, 97, 126, 162, 204, 254, 317, 356))
ref_deltas = np.asarray(("80-100%", "70-80%", "60-70%", "50-60%", "40-50%",
                         "30-40%", "20-30%", "10-20%", "5-10%", "0-5%"))
if use_my_files is True:
    runs, pro_int_ave, protons, refmult, net_protons = fn.protons_my_data(low_bound=100,
                                                                          high_bound=300,
                                                                          run_plot=False,
                                                                          std_mag=2)
    runs1, pro_int_ave1, protons1, ref_raw, pro_raw = fn.protons_my_data(low_bound=0,
                                                                         high_bound='max',
                                                                         run_plot=False,
                                                                         std_mag=2)
    index = (refmult >= 1) & (refmult <= 2000)
    refmult = refmult[index]
    net_protons = net_protons[index]
    # refmult = ak.to_numpy(ak.flatten(protons[0]))
    # net_protons = ak.to_numpy(ak.flatten(protons[3]))
    # fn.plot_pro_ave(runs, pro_int_ave)
    # fn.plot_pro_dist_refmult(protons)

elif use_yu_files is True:
    file = r'C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results\moments.root'
    ref_raw, pro_raw = fn.protons_yu_data(data_cut=1)
    refmult, net_protons = fn.protons_yu_data(data_cut=4)
    index = (refmult >= 1) & (refmult <= 2000)
    refmult = refmult[index]
    net_protons = net_protons[index]

elif use_ml_sets is True:
    loc = r"C:\Users\dansk\Documents\Thesis\2021_DNP_Work\ML_fits\9_2021\\"
    refmult, predictions = fn.protons_ml_data(loc=loc)
    ref_nums = [0, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    ref_nums = [0, 12.9, 23.9, 36.5, 47.3, 58.2, 69.1, 79.9, 90.6, 95.6]
    r = len(refmult)
    file = r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results\BigList\out_all.pkl"
    a, b, net_protons = fn.protons_my_pandas(file, data_cut=4)
    index = (refmult >= 1) & (refmult <= 2000)
    ring_sum = predictions[0]
    ring_sum_outer = predictions[1]
    pred_l = predictions[2]
    pred_r = predictions[3]
    pred_s = predictions[4]
    pred_m = predictions[5]
    preds = np.array((ring_sum, ring_sum_outer, pred_l, pred_r, pred_s, pred_m))

    fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    labels = [r"$\Sigma EPD$", r"$\Sigma EPD_{out}$", r"$X_{\zeta',LW}$",
              r"$X_{\zeta',RELU}$", r"$X_{\zeta',swish}$", r"$X_{\zeta',mish}$"]
    for i in range(2):
        for j in range(3):
            x = i*3 + j
            print(labels[x], ":", np.corrcoef(refmult, preds[x])[0][1])
            count, binsX, binsY = np.histogram2d(refmult, preds[x], bins=700, range=((0, 700), (0, 700)))
            im = ax[i, j].pcolormesh(binsX, binsY, count, cmap='jet', norm=LogNorm(), shading='auto')
            fig.colorbar(im, ax=ax[i, j])
            ax[i, j].set_ylabel(labels[x], fontsize=20)
            ax[i, j].set_xlabel("RefMult3", fontsize=20)
    plt.show()

    rsum_cuts = []
    pred_after = []
    for i in range(6):
        pred_after.append(preds[i][index])
        rsum_cuts.append(np.percentile(preds[i], ref_nums))
    pred_after = np.array(pred_after).astype(int)
    refmult = refmult[index]
    ref_cuts = np.percentile(refmult, ref_nums)
    net_protons = net_protons[index]

else:
    file = r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results\BigList\out_all.pkl"
    ref_raw, rsum_raw, pro_raw = fn.protons_my_pandas(file, data_cut=1)
    refmult, rsum, net_protons = fn.protons_my_pandas(file, data_cut=8)
    index = (refmult >= 1) & (refmult <= 2000)
    refmult = refmult[index]
    rsum = rsum[index]
    net_protons = net_protons[index]
    rsum = rsum.astype("int")

    # species_energy_n = "197Au_197Au_14.5_100000"
    # gmc_file = r"C:\Users\dansk\Documents\Thesis\Tristan"
    # n_part = np.load(r'{}\Npart_{}.npy'.format(gmc_file, species_energy_n), allow_pickle=True)
    # n_coll = np.load(r'{}\Ncoll_{}.npy'.format(gmc_file, species_energy_n), allow_pickle=True)
    # gmc = fn.gmc_dist_generator(n_coll, n_part, 1.2, 0.53, 0)
    # bins = 1800
    # scaler_rings, x = np.histogram(rsum, bins=bins, range=(0, bins))
    # x = x[:-1]
    # match = 363
    # percent_nums = [20, 30, 40, 50, 60, 70, 80, 90, 95]

# # fn.proton_histos_1d(refmult, net_protons, ref_raw, pro_raw, ref_cuts, ref_deltas)
# This will give us the cumulants in both forms.

pro_int_cumus_pred = []
pro_int_cumcomp_pred = []
pro_int_errs_pred = []
pro_len_tot_pred = []
pro_len_tot_pred_err = []
pro_len_tot_comp_pred = []
for i in range(6):
    pro_int_cumus_epd, pro_int_cumcomp_epd, pro_int_errs_epd = fn.int_cumulants_protons(refmult=pred_after[i],
                                                                                        protons=net_protons)
    pro_int_cumus_pred.append(pro_int_cumus_epd)
    pro_int_cumcomp_pred.append(pro_int_cumcomp_epd)
    pro_int_errs_pred.append(pro_int_errs_epd)
    pro_len_tot_epd, pro_len_tot_comp_epd, pro_len_tot_epd_err = fn.apply_cbwc_protons(pred_after[i], net_protons, ref_cuts=rsum_cuts[i])
    pro_len_tot_pred.append(pro_len_tot_epd)
    pro_len_tot_comp_pred.append(pro_len_tot_comp_epd)
    pro_len_tot_pred_err.append(pro_len_tot_epd_err)
# pro_int_cumus_pred = np.array(pro_int_cumus_pred)
# pro_int_cumcomp_pred = np.array(pro_int_cumcomp_pred)
# pro_int_errs_pred = np.array(pro_int_errs_pred)
# pro_len_tot_pred = np.array(pro_len_tot_pred)
# pro_len_tot_comp_pred = np.array(pro_len_tot_comp_pred)

pro_int_cumus, pro_int_cumcomp, pro_int_errs = fn.int_cumulants_protons(refmult=refmult, protons=net_protons)

# pro_int_cumus_epd, pro_int_cumcomp_epd, pro_int_errs_epd = fn.int_cumulants_protons(refmult=rsum, protons=net_protons)
# This applies CBWC
pro_len_tot, pro_len_tot_comp, pro_len_tot_err = fn.apply_cbwc_protons(refmult, net_protons, ref_cuts=ref_cuts)

# pro_len_tot_epd, pro_len_tot_comp_epd = fn.apply_cbwc_protons(rsum, net_protons, ref_cuts=rsum_cuts)
"""
max_ref = int(np.max(refmult)*1.2)
max_pro = int(np.max(net_protons)*1.2)
counter, binsX, binsY = np.histogram2d(refmult, net_protons,
                                       bins=(max_ref, max_pro), range=((0, max_ref), (0, max_pro)))
"""
"""
################## PLOTS ######################
"""
# Plot of the net proton cumulants, by RefMult.
names = [r"$\Sigma EPD$", r"$\Sigma EPD_{out}$", r"$X_{\zeta,LW}$",
         r"$X_{\zeta,RELU}$", r"$X_{\zeta,swish}$", r"$X_{\zeta,mish}$"]
fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
for i in range(2):
    for j in range(2):
        x = 2 * i + j
        ax[i, j].errorbar(np.unique(refmult), pro_int_cumus[x], yerr=pro_int_errs[x],
                          lw=0, marker='.', elinewidth=1, capsize=2, ms=2, alpha=0.5,
                          label='refmult')
        for k in range(6):
            ax[i, j].errorbar(np.unique(pred_after[k]), pro_int_cumus_pred[k][x], yerr=pro_int_errs_pred[k][x],
                              lw=0, marker='.', elinewidth=1, capsize=2, ms=2, alpha=0.5,
                              label=names[k])
        ax[i, j].legend()
plt.suptitle("WOOP WOP TACO TRUCK!!!", fontsize=30)
plt.show()
plt.close()
"""
# Plot of the net proton cumulant ratios, by RefMult.
fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
for i in range(2):
    for j in range(2):
        x = 2 * i + j
        ax[i, j].plot(np.unique(refmult), pro_int_cumcomp[x])
plt.show()
plt.close()
"""
# Plot of the CBWC cumulants.
cumulants = [r"$C_1$", r"$C_2$", r"$C_3$", r"$C_4$"]
marker = ["v", "^", "<", ">", "P", "X"]
fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
for i in range(2):
    for j in range(2):
        x = 2 * i + j
        ax[i, j].errorbar(ref_deltas, pro_len_tot[x], pro_len_tot_err[x], lw=0, marker="*", ms=10, c='red', alpha=0.5,
                          label="RefMult3")
        for k in range(6):
            ax[i, j].errorbar(ref_deltas, pro_len_tot_pred[k][x], pro_len_tot_pred_err[k][x],
                              lw=0, marker=marker[k], elinewidth=1,
                              capsize=2,  ms=10, alpha=0.5,
                              label=names[k])
        ax[i, j].set_title(cumulants[x], fontsize=25)
        if x == 3:
            ax[i, j].set_ylim(-150, 350)
        ax[i, j].legend()
plt.show()
plt.close()

# And of the CBWC ratios.
cum_ratios = [r"$\mu$", r"$\frac{\sigma^2}{\mu}$", r"$S\sigma$", r"$\kappa\sigma^2$"]
fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
for i in range(2):
    for j in range(2):
        x = 2 * i + j
        ax[i, j].plot(ref_deltas, pro_len_tot_comp[x], lw=0, marker="*", ms=10, c='red', alpha=0.5,
                      label="RefMult3")
        for k in range(6):
            ax[i, j].plot(ref_deltas, pro_len_tot_comp_pred[k][x], lw=0, marker=marker[k], ms=10, alpha=0.5,
                          label=names[k])
        ax[i, j].set_title(cum_ratios[x], fontsize=25)
        ax[i, j].legend()
plt.show()
plt.close()
