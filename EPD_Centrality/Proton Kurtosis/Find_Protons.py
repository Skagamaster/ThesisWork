import os
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# import functions as fn
# from scipy.optimize import curve_fit
# import pandas as pd
# 6565234
# These are the non cbwc values (in centrality selections).
os.chdir(r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results")
protons = np.load("protons.npy", allow_pickle=True)

# These are the protons for cbwc.
os.chdir(r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results")
proton_dists = np.load("proton_sum.npy", allow_pickle=True)

# And now for the rest of the values.
ave_protons = np.load("Ave_protons.npy", allow_pickle=True)
prot_ave = np.mean(ave_protons[0])
prot_std = 3 * np.std(ave_protons[0])
dEdXpG = np.load("dEdX_pq.npy", allow_pickle=True)
dEdXpGbins = np.load("dEdX_pq_bins.npy", allow_pickle=True)
runs = np.load("run_list.npy", allow_pickle=True)

# 2D histogram data.
v_r = np.load("vR.npy", allow_pickle=True)
v_r_bins = np.load("vR_bins.npy", allow_pickle=True)
RefMult_TOFMult = np.load("refmult_tofmult.npy", allow_pickle=True)
RefMult_TOFMult_bins = np.load("refmult_tofmult_bins.npy", allow_pickle=True)
RefMult_TOFMatch = np.load("refmult_tofmatch.npy", allow_pickle=True)
RefMult_TOFMatch_bins = np.load("refmult_tofmatch_bins.npy", allow_pickle=True)
RefMult_BetaEta = np.load("refmult_beta_eta.npy", allow_pickle=True)
RefMult_BetaEta_bins = np.load("refmult_beta_eta_bins.npy", allow_pickle=True)
m_pq = np.load("m_pq.npy", allow_pickle=True)
m_pq_bins = np.load("m_pq_bins.npy", allow_pickle=True)
beta_p = np.load("beta_p.npy", allow_pickle=True)
beta_p_bins = np.load("beta_p_bins.npy", allow_pickle=True)
dEdX_pq = np.load("dEdX_pq.npy", allow_pickle=True)
dEdX_pq_bins = np.load("dEdX_pq_bins.npy", allow_pickle=True)

# RefMult3 found by Yu (similar to my quantiles)
Ref3Cuts = np.asarray((0, 10, 21, 41, 72, 118, 182, 270, 392, 472))
# If we're just plotting by centrality %.
RefCuts = np.linspace(1, 10, 10)
RefCuts_labels = ["0-5%", "5-10%", "10-20%", "20-30%", "30-40%", "40-50%",
                  "50-60%", "60-70%", "70-80%", "80-100%"]

# Cumulant analysis.
cumulant_names = np.asarray(("Mean", "Variance", "Skewness", "Kurtosis"))
# Cumulants for centrality ranges, but with no CBWC.
cumulants = np.zeros((4, len(protons)))
for i in range(len(protons)):
    cumulants[0][i] = np.mean(protons[i])
    cumulants[1][i] = np.var(protons[i])
    cumulants[2][i] = skew(protons[i]) * np.power(np.sqrt(np.var(protons[i])), 3)
    cumulants[3][i] = kurtosis(protons[i]) * np.power(np.var(protons[i]), 2)
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

# Plot for the average protons per run.
plt.figure(figsize=(12, 9))
x = np.linspace(0, len(ave_protons[0]), len(ave_protons[0]))
plt.errorbar(runs, ave_protons[0], yerr=ave_protons[1], fmt='ok', ms=0,
             mfc='None', capsize=1.5, elinewidth=1)
plt.xticks(runs[::10])
plt.tick_params('x', labelrotation=90, labelsize=7)
plt.hlines(prot_ave + prot_std, np.min(x), np.max(x), colors='blue', label=r'3$\sigma$', linestyle="dashed")
plt.hlines(prot_ave - prot_std, np.min(x), np.max(x), colors='blue', linestyle="dashed")
plt.hlines(prot_ave, np.min(x), np.max(x), colors="red", linestyle="dashed", label="ave")
plt.xlabel("RunID", fontsize=20)
plt.ylabel("<Protons>", fontsize=20)
plt.legend()
plt.title("Average Proton Count per Run", fontsize=30)
plt.show()
plt.close()

# Plot for the cumulants.
fig, ax = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

# Yu's cumulant values (CBWC, uncorrected for efficiencies).
C = [[0, 0.5, 0.95, 1.58, 2.88, 4.61, 7.09, 10.59, 14.0, 16.78],
     [0, 0.17, 0.78, 1.65, 2.99, 4.85, 7.53, 11.26, 14.89, 17.79],
     [0, 0.17, 0.63, 1.30, 2.39, 3.87, 6.10, 9.10, 11.96, 13.79],
     [0, 0.51, 1.10, 1.78, 2.54, 4.24, 6.53, 9.75, 13.05, 15.34]]

for i in range(2):
    for j in range(2):
        x = i * 2 + j
        ax[i, j].plot(RefCuts, np.asarray(cumulants_cbwc[x]), c='b', marker="o", mfc='red', lw=0, mew=2,
                      label="Me_cbwc", alpha=0.5)
        ax[i, j].plot(RefCuts, np.asarray(C[x]), marker="o", c='orange', mfc='green', lw=0, mew=2, label="Yu", alpha=0.5)
        ax[i, j].set_title(cumulant_names[x])
        ax[i, j].set_xticks(RefCuts[::-1])
        ax[i, j].set_xticklabels(RefCuts_labels, rotation=45)
        ax[i, j].set_xlabel("Centrality")
        ax[i, j].grid(True)
        ax[i, j].legend()
fig.suptitle("Net Proton Cumulants: Mine v Yu", fontsize=20)
plt.show()
plt.close()

fig, ax = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

for i in range(2):
    for j in range(2):
        x = i * 2 + j
        ax[i, j].plot(RefCuts, cumulants_cbwc[x], c='b', marker="o", mfc='red', lw=0, mew=2,
                      label="CBWC", alpha=1.0)
        ax[i, j].plot(RefCuts, cumulants[x][::-1], marker="o", c='black', mfc='pink', lw=0, mew=2,
                      label="No CBWC", alpha=1.0)
        ax[i, j].set_title(cumulant_names[x])
        ax[i, j].set_xticks(RefCuts[::-1])
        ax[i, j].set_xticklabels(RefCuts_labels, rotation=45)
        ax[i, j].set_xlabel("Centrality")
        ax[i, j].grid(True)
        ax[i, j].legend()
fig.suptitle("Net Proton Cumulants: With and without CBWC", fontsize=20)
plt.show()
plt.close()

''' Turned off for now.
# Plot of dE/dX vs pq, for reference.
plt.figure(figsize=(12, 9))
X, Y = np.meshgrid(dEdXpGbins[1], dEdXpGbins[0])
plt.pcolormesh(X, Y, dEdXpG, cmap="jet", norm=colors.LogNorm())
plt.title(r"$\frac{dE}{dx}$ vs $p_G$, After Track Cuts", fontsize=30)
plt.xlabel(r"$p_G*q (\frac{GeV}{c})$", fontsize=20)
plt.ylabel(r"$\frac{dE}{dx} (\frac{KeV}{cm})$", fontsize=20)
plt.colorbar()
plt.xlim([-5, 5])
plt.ylim([0, 20])
plt.tight_layout()
plt.show()
plt.close()
'''

# 2D histograms
fig, ax = plt.subplots(2, 4, figsize=(16, 9), constrained_layout=True)

# Plot for RefMult3 vs TOFMult
x = np.linspace(0, 1000, 1001)
y1 = 1.22 * x - 24.29
y2 = 1.95 * x + 75
ax[0, 0].plot(x, y1, c='red', lw=2)
ax[0, 0].plot(x, y2, c='red', lw=2)
X, Y = np.meshgrid(RefMult_TOFMult_bins[1], RefMult_TOFMult_bins[0])
imRefMult_TOFMult = ax[0, 0].pcolormesh(X, Y, RefMult_TOFMult, cmap="jet", norm=colors.LogNorm())
ax[0, 0].set_xlabel("RefMult3", fontsize=10)
ax[0, 0].set_ylabel("TOFMult", fontsize=10)
ax[0, 0].set_title("RefMult3 vs TOFMult", fontsize=20)
ax[0, 0].set_ylim(0, np.max(Y))
fig.colorbar(imRefMult_TOFMult, ax=ax[0, 0])

# Plot for RefMult3 vs TOFMatch
y1 = 0.379 * x - 8.6
y2 = 0.631 * x + 11.69
ax[0, 1].plot(x, y1, c='red', lw=2)
ax[0, 1].plot(x, y2, c='red', lw=2)
X, Y = np.meshgrid(RefMult_TOFMatch_bins[1], RefMult_TOFMatch_bins[0])
imRefMult_TOFMatch = ax[0, 1].pcolormesh(X, Y, RefMult_TOFMatch, cmap="jet", norm=colors.LogNorm())
ax[0, 1].set_xlabel("RefMult3", fontsize=10)
ax[0, 1].set_ylabel("TOFMatch", fontsize=10)
ax[0, 1].set_title("RefMult3 vs TOFMatch", fontsize=20)
ax[0, 1].set_ylim(0, np.max(Y))
fig.colorbar(imRefMult_TOFMatch, ax=ax[0, 1])

# Plot for RefMult3 vs beta_eta1
y1 = 0.417 * x - 13.1
y2 = 0.526 * x + 14
ax[0, 2].plot(x, y1, c='red', lw=2)
ax[0, 2].plot(x, y2, c='red', lw=2)
X, Y = np.meshgrid(RefMult_BetaEta_bins[1], RefMult_BetaEta_bins[0])
imRefMult_BetaEta = ax[0, 2].pcolormesh(X, Y, RefMult_BetaEta, cmap="jet", norm=colors.LogNorm())
ax[0, 2].set_xlabel("RefMult3", fontsize=10)
ax[0, 2].set_ylabel(r"$\beta \eta$1", fontsize=10)
ax[0, 2].set_title(r"RefMult3 vs $\beta \eta$1", fontsize=20)
ax[0, 2].set_ylim(0, np.max(Y))
fig.colorbar(imRefMult_BetaEta, ax=ax[0, 2])

# Plot for v_x vs v_y
# Centering the circle on the most probable location:
# circle1 = plt.Circle((-0.27275,-0.235275),2, color = 'red', fill = False, clip_on = False, lw=2)
# Centering the circle on the origin:
circle1 = plt.Circle((0, 0), 2, color='red', fill=False, clip_on=False, lw=2)
X, Y = np.meshgrid(v_r_bins[1], v_r_bins[0])
imVr = ax[0, 3].pcolormesh(X, Y, v_r, cmap="jet", norm=colors.LogNorm())
ax[0, 3].set_xlabel(r"$v_x$ (cm)", fontsize=10)
ax[0, 3].set_ylabel(r"$v_y$ (cm)", fontsize=10)
ax[0, 3].set_title(r"$v_r$ Position", fontsize=20)
ax[0, 3].add_artist(circle1)
# fig.colorbar(imVr, ax=ax[0, 3])

# Plot for mass vs momentum
X, Y = np.meshgrid(m_pq_bins[1], m_pq_bins[0])
imMp = ax[1, 0].pcolormesh(X, Y, m_pq, cmap="jet", norm=colors.LogNorm())
ax[1, 0].set_ylabel(r"$m^2 (\frac{GeV^2}{c^4})$", fontsize=10)
ax[1, 0].set_xlabel(r"p*q $(\frac{GeV}{c})$", fontsize=10)
ax[1, 0].set_title("Mass vs Momentum", fontsize=20)
ax[1, 0].axhline(0.6, c='red', lw=2)
ax[1, 0].axhline(1.2, c='red', lw=2)
fig.colorbar(imMp, ax=ax[1, 0])

# Plot for Beta vs momentum
X, Y = np.meshgrid(beta_p_bins[1], beta_p_bins[0])
imBp = ax[1, 1].pcolormesh(X, Y, beta_p, cmap="jet", norm=colors.LogNorm())
ax[1, 1].set_xlabel(r"p $(\frac{GeV}{c})$", fontsize=10)
ax[1, 1].set_ylabel(r"1/$\beta$", fontsize=10)
ax[1, 1].set_title(r"$\beta$ vs Momentum", fontsize=20)
fig.colorbar(imBp, ax=ax[1, 1])

# Plot for dEdX vs momentum
X, Y = np.meshgrid(dEdX_pq_bins[1], dEdX_pq_bins[0])
imDp = ax[1, 2].pcolormesh(X, Y, dEdX_pq, cmap="jet", norm=colors.LogNorm())
ax[1, 2].set_xlabel(r"p*q $(\frac{GeV}{c})$", fontsize=10)
ax[1, 2].set_ylabel(r"$\frac{dE}{dX} (\frac{KeV}{cm})$", fontsize=10)
ax[1, 2].set_title(r"p*q vs $\frac{dE}{dX}$", fontsize=20)
fig.colorbar(imDp, ax=ax[1, 2])

ax[1, 3].set_axis_off()

plt.show()
plt.close()

# Get the list of bad runs from proton averages.
os.chdir(r"C:\Users\dansk\Documents\Thesis\Protons\WIP\Protons")
bad_high = prot_ave + prot_std
bad_low = prot_ave - prot_std
badlist_high = np.hstack(np.where(ave_protons[0] > bad_high))
badlist_low = np.hstack(np.where(ave_protons[0] < bad_low))
badlist = [runs[badlist_high], runs[badlist_low]]
badlist = np.hstack(np.asarray(badlist))
#np.save("badlist.npy", badlist)
