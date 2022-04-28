#
# \Just a little something to look at my QA data with.
#
# \author Skipper KAgamaster
# \date 04/11/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
# /

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema as arex
from scipy.signal import savgol_filter as sgf
from scipy.stats import skew, kurtosis
import uproot as up
import awkward as ak
import time
import logging

load_direct = r"C:\Users\dansk\Documents\Thesis\Protons\WIP\Protons\YuCuts\Histograms"
os.chdir(load_direct)

averages = pd.read_pickle("averages.pkl")
ave_len = len(averages)
ave_len = np.linspace(0, ave_len-1, ave_len)
columns = ['ave_refmult3', 'err_refmults', 'ave_vz', 'err_vz', 'ave_vr', 'err_vr', 'ave_pt', 'err_pt', 'ave_eta',
           'err_eta', 'ave_zdcx', 'err_zdcx', 'ave_phi', 'err_phi', 'ave_dca', 'err_dca']
ave_titles = [r"$RefMult3$", r"$v_z$", r"$v_r$", r"$p_T$", r"$\eta$", r"$ZDC_x$", r"$\phi$", r"$DCA$"]
ave_ylabels = [r"<$RefMult3$>", r"<$v_z$> (cm)", r"<$v_r$> (cm)", r"<$p_T> (\frac{GeV}{c})$",
               r"<$\eta$>", r"<$ZDC_x$> (cm)", r"<$\phi$> (rad)", r"<$DCA$> (cm)"]

# Let's make some plots!
fig, ax = plt.subplots(2, 4, figsize=(16, 9), constrained_layout=True)
for i in range(2):
    for j in range(4):
        x = i*4 + j
        ave = np.mean(averages[columns[x*2]])
        std = 3*np.std(averages[columns[x*2]])
        ax[i, j].errorbar(ave_len, averages[columns[2*x]].to_numpy(), yerr=averages[columns[2*x + 1]].to_numpy(),
                          fmt='ok', ms=0, mfc='None', capsize=1.5, elinewidth=1)
        ax[i, j].set_xticks(ax[0, 0].get_xticks()[::100])
        ax[i, j].tick_params(labelrotation=45, labelsize=7)
        ax[i, j].axhline(ave, 0, 1, c='red', ls="--", label="Mean")
        ax[i, j].axhline(ave+std, 0, 1, c='blue', ls="--", label=r'3$\sigma$')
        ax[i, j].axhline(ave-std, 0, 1, c='blue', ls="--")
        ax[i, j].set_title(ave_titles[x], fontsize=20)
        ax[i, j].set_ylabel(ave_ylabels[x], fontsize=10)
        ax[i, j].legend()
plt.show()

# Now to plot the histograms for my before data. I need to OOP this noise.
# 1D histograms
vz_hist = np.load("vz_hist.npy", allow_pickle=True)
dca_hist = np.load("dca_hist.npy", allow_pickle=True)
eta_hist = np.load("eta_hist.npy", allow_pickle=True)
nhits_dedx_hist = np.load("nhits_dedx_hist.npy", allow_pickle=True)
nhitsq_hist = np.load("nhitsq_hist.npy", allow_pickle=True)
phi_hist = np.load("phi_hist.npy", allow_pickle=True)
pt_hist = np.load("pt_hist.npy", allow_pickle=True)
ref_hist = np.load("ref_hist.npy", allow_pickle=True)

fig, ax = plt.subplots(2, 4, figsize=(16, 9), constrained_layout=True)

ax[0, 0].plot(pt_hist[1][:-1], pt_hist[0])
ax[0, 0].set_xlabel(r"$p_T$ $(\frac{GeV}{c})$")
ax[0, 0].set_ylabel("Counts")
ax[0, 0].set_yscale('log')

ax[0, 1].plot(phi_hist[1][:-1], phi_hist[0])
ax[0, 1].set_xlabel(r"$\phi$")
ax[0, 1].set_ylabel("Counts")
ax[0, 1].set_yscale('log')

ax[0, 2].plot(dca_hist[1][:-1], dca_hist[0])
ax[0, 2].set_xlabel("DCA (cm)")
ax[0, 2].set_ylabel("Counts")
ax[0, 2].set_yscale('log')

ax[0, 3].plot(ref_hist[1][:-1], ref_hist[0])
ax[0, 3].set_xlabel("RefMult3")
ax[0, 3].set_ylabel("Counts")
ax[0, 3].set_yscale('log')

ax[1, 0].plot(eta_hist[1][:-1], eta_hist[0])
ax[1, 0].set_xlabel(r"$\eta$")
ax[1, 0].set_ylabel("Counts")
ax[1, 0].set_yscale('log')

ax[1, 1].plot(nhitsq_hist[1][:-1], nhitsq_hist[0])
ax[1, 1].set_xlabel("nHitsFit*charge")
ax[1, 1].set_ylabel("Counts")
ax[1, 1].set_yscale('log')

ax[1, 2].plot(nhits_dedx_hist[1][:-1], nhits_dedx_hist[0])
ax[1, 2].set_xlabel("nHits_dEdX")
ax[1, 2].set_ylabel("Counts")
ax[1, 2].set_yscale('log')

ax[1, 3].plot(vz_hist[1][:-1], vz_hist[0])
ax[1, 3].set_xlabel(r"$v_z$ (cm)")
ax[1, 3].set_ylabel("Counts")
ax[1, 3].set_yscale('log')

plt.show()
plt.close()

# 2D histograms
vr_hist = np.load("vr_hist.npy", allow_pickle=True)
betap_hist = np.load("betap_hist.npy", allow_pickle=True)
dedx_pq_hist = np.load("dedx_pq_hist.npy", allow_pickle=True)
mpq_hist = np.load("mpq_hist.npy", allow_pickle=True)
ref_beta_hist = np.load("ref_beta_hist.npy", allow_pickle=True)
rt_match_hist = np.load("rt_match_hist.npy", allow_pickle=True)
rt_mult_hist = np.load("rt_mult_hist.npy", allow_pickle=True)

fig, ax = plt.subplots(2, 4, figsize=(16, 9), constrained_layout=True)

# Plot for RefMult3 vs TOFMult
"""
x = np.linspace(0, 1000, 1001)
y1 = 1.22*x - 24.29
y2 = 1.95*x + 75
ax[0, 0].plot(x, y1, c='red', lw=2)
ax[0, 0].plot(x, y2, c='red', lw=2)
"""
X, Y = np.meshgrid(rt_mult_hist[2], rt_mult_hist[1])
imRefMult_TOFMult = ax[0, 0].pcolormesh(X, Y, rt_mult_hist[0], cmap="jet", norm=colors.LogNorm())
ax[0, 0].set_xlabel("RefMult3", fontsize=10)
ax[0, 0].set_ylabel("TOFMult", fontsize=10)
ax[0, 0].set_title("RefMult3 vs TOFMult", fontsize=20)
ax[0, 0].set_ylim(0, np.max(Y))
fig.colorbar(imRefMult_TOFMult, ax=ax[0, 0])

# Plot for RefMult3 vs TOFMatch
"""
y1 = 0.379*x - 8.6
y2 = 0.631*x + 11.69
ax[0, 1].plot(x, y1, c='red', lw=2)
ax[0, 1].plot(x, y2, c='red', lw=2)
"""
X, Y = np.meshgrid(rt_match_hist[2], rt_match_hist[1])
imRefMult_TOFMatch = ax[0, 1].pcolormesh(X, Y, rt_match_hist[0], cmap="jet", norm=colors.LogNorm())
ax[0, 1].set_xlabel("RefMult3", fontsize=10)
ax[0, 1].set_ylabel("TOFMatch", fontsize=10)
ax[0, 1].set_title("RefMult3 vs TOFMatch", fontsize=20)
ax[0, 1].set_ylim(0, np.max(Y))
fig.colorbar(imRefMult_TOFMatch, ax=ax[0, 1])

# Plot for RefMult3 vs beta_eta1
"""
y1 = 0.202*x - 11.5
y2 = 0.37*x + 8
ax[0, 2].plot(x, y1, c='red', lw=2)
ax[0, 2].plot(x, y2, c='red', lw=2)
"""
X, Y = np.meshgrid(ref_beta_hist[2], ref_beta_hist[1])
imRefMult_BetaEta = ax[0, 2].pcolormesh(X, Y, ref_beta_hist[0], cmap="jet", norm=colors.LogNorm())
ax[0, 2].set_xlabel("RefMult3", fontsize=10)
ax[0, 2].set_ylabel(r"$\beta \eta$1", fontsize=10)
ax[0, 2].set_title(r"RefMult3 vs $\beta \eta$1", fontsize=20)
ax[0, 2].set_ylim(0, np.max(Y))
fig.colorbar(imRefMult_BetaEta, ax=ax[0, 2])

# Plot for v_x vs v_y
# circle1 = plt.Circle((-0.27275,-0.235275),2, color = 'red', fill = False, clip_on = False, lw=2)
X, Y = np.meshgrid(vr_hist[2], vr_hist[1])
imVr = ax[0, 3].pcolormesh(X, Y, vr_hist[0], cmap="jet", norm=colors.LogNorm())
ax[0, 3].set_xlabel(r"$v_x$ (cm)", fontsize=10)
ax[0, 3].set_ylabel(r"$v_y$ (cm)", fontsize=10)
ax[0, 3].set_title(r"$v_r$ Position", fontsize=20)
# ax[0, 3].add_artist(circle1)
# fig.colorbar(imVr, ax=ax[0, 3])

# Plot for mass vs momentum
X, Y = np.meshgrid(mpq_hist[2], mpq_hist[1])
imMp = ax[1, 0].pcolormesh(X, Y, mpq_hist[0], cmap="jet", norm=colors.LogNorm())
ax[1, 0].set_ylabel(r"$m^2 (\frac{GeV^2}{c^4})$", fontsize=10)
ax[1, 0].set_xlabel(r"p*q $(\frac{GeV}{c})$", fontsize=10)
ax[1, 0].set_title("Mass vs Momentum", fontsize=20)
# ax[1, 0].axhline(0.6, c='red', lw=2)
# ax[1, 0].axhline(1.2, c='red', lw=2)
fig.colorbar(imMp, ax=ax[1, 0])

# Plot for Beta vs momentum
X, Y = np.meshgrid(betap_hist[2], betap_hist[1])
imBp = ax[1, 1].pcolormesh(X, Y, betap_hist[0], cmap="jet", norm=colors.LogNorm())
ax[1, 1].set_xlabel(r"p $(\frac{GeV}{c})$", fontsize=10)
ax[1, 1].set_ylabel(r"1/$\beta$", fontsize=10)
ax[1, 1].set_title(r"$\beta$ vs Momentum", fontsize=20)
fig.colorbar(imBp, ax=ax[1, 1])

# Plot for dEdX vs momentum
X, Y = np.meshgrid(dedx_pq_hist[2], dedx_pq_hist[1])
imDp = ax[1, 2].pcolormesh(X, Y, dedx_pq_hist[0], cmap="jet", norm=colors.LogNorm())
ax[1, 2].set_xlabel(r"p*q $(\frac{GeV}{c})$", fontsize=10)
ax[1, 2].set_ylabel(r"$\frac{dE}{dX} (\frac{KeV}{cm})$", fontsize=10)
ax[1, 2].set_title(r"p*q vs $\frac{dE}{dX}$", fontsize=20)
fig.colorbar(imDp, ax=ax[1, 2])

ax[1, 3].set_axis_off()

plt.show()
