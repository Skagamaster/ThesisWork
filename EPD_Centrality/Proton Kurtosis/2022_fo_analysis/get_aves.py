# \Find average quantities in FastOffline data
# \to make bad run list.
#
#
# \author Skipper Kagamaster
# \date 04/22/2022
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

'''
The quantities in the imported array are as follows:
arr[0] = runID, arr[1] = N_events, arr[2] = N_tracks
arr[3] = <RefMult3>, arr[4] = <v_z>, arr[5] = <v_r>
arr[6] = <phi>, arr[7] = <p_T>, arr[8] = <ZDC_x>, arr[9] = <eta>, arr[10] = <DCA>
arr[11:42] = <RingSum[i-11]> (EPD rings 0-31)
arr[43:50] = var[i-40] (starts at RefMult3), arr[51:82] = var[RingSum[i-51]]
'''

os.chdir(r'D:\14GeV\Stds_Picos')
aves = []
stds = []
e_set = [3, 4, 5]
t_set = [6, 7, 8, 9, 10]
emptyruns = []
for i in range(32):
    e_set.append(i + 11)
ve_set = [43, 44, 45]
vt_set = [46, 47, 48, 49, 50]
for i in range(32):
    ve_set.append(i + 51)
for i in os.listdir():
    if i.endswith('txt'):
        arr = np.loadtxt(i)
        if arr[2] > 100:
            stds.append(arr[43:])
            # This is to turn var into SEM.
            # --------------------------------------------------- #
            arr[43:] = np.sqrt(arr[43:])
            arr[43:46] = np.divide(arr[43:46], np.sqrt(arr[1]))
            arr[46:51] = np.divide(arr[46:51], np.sqrt(arr[2]))
            arr[51:] = np.divide(arr[51:], np.sqrt(arr[1]))
            # --------------------------------------------------- #
            aves.append(arr)
        else:
            emptyruns.append(i[4:12])  # We'll add this to the badrun list after plotting.
            continue
aves = np.asarray(aves).T
stds = np.asarray(stds).T

runs = aves[0].astype('int')
print("Number of picos:", len(runs))
nevents = aves[1].astype('int')
ntracks = aves[2].astype('int')
print("Total events:", np.sum(nevents))
print("Total tracks:", np.sum(ntracks))
x_ax = np.linspace(1, len(runs), len(runs))
tot_ave = np.empty(40)
tot_std = np.empty(40)
# Now to find the average and std over all runs.
# We'll use weighted averages for this.
for i in range(40):
    if i + 3 in e_set:
        wave = np.average(aves[i + 3], weights=nevents / np.power(stds[i], 2))
        tot_ave[i] = wave
        tot_std[i] = np.sqrt(np.average((aves[i + 3] - wave) ** 2, weights=nevents / np.power(stds[i], 2)))
    elif i + 3 in t_set:
        if i + 3 == 8:
            wave = np.average(aves[i + 3], weights=ntracks)
            tot_std[i] = np.sqrt(np.average((aves[i + 3] - wave) ** 2, weights=ntracks))
        else:
            wave = np.average(aves[i + 3], weights=ntracks / np.power(stds[i], 2))
            tot_std[i] = np.sqrt(np.average((aves[i + 3] - wave) ** 2, weights=ntracks / np.power(stds[i], 2)))
        tot_ave[i] = wave

tot_std = tot_std * 4

high_point = tot_ave + tot_std
low_point = tot_ave - tot_std
badruns = []
for i in range(40):
    wave_up = aves[i + 3] + stds[i]
    wave_down = aves[i + 3] - stds[i]
    index = np.where(((wave_up > high_point[i]) | (wave_down < low_point[i])))
    badruns.append(runs[index])
badruns = np.hstack(badruns)
badruns = np.unique(badruns)

print("Percentage of bad runs:", np.round(len(badruns) / len(runs), 4) * 100, "%")
badargs = []
goodargs = []
for i in range(len(runs)):
    if runs[i] in badruns:
        badargs.append(i)
    else:
        goodargs.append(i)
goodruns = np.asarray(runs[goodargs])
badruns = np.asarray(runs[badargs])

print("Total good runs:", len(goodruns))
print("Total events in good runs:", np.sum(nevents[goodargs]))
print("Total tracks in good runs:", np.sum(ntracks[goodargs]))

y_labels = ['<RefMult3>', r"<$v_z$> (cm)", r"<$v_r$> (cm)", r"<$\phi$>", r"<$p_T$> ($\frac{GeV}{c}$)",
            r"<$ZDC_x$>", r"<$\eta$>", "<DCA> (cm)"]
fig, ax = plt.subplots(3, 3, figsize=(12, 7), constrained_layout=True)
for i in range(3):
    for j in range(3):
        x = i * 3 + j + 3
        if x > 10:
            ax[i, j].set_axis_off()
            continue
        elif x == 8:
            ax[i, j].set_axis_off()
            continue
        ax[i, j].errorbar(x_ax, aves[x], yerr=aves[x + 40], fmt='ok', ms=0,
                          mfc='None', capsize=1.5, elinewidth=1, label='good runs')
        ax[i, j].errorbar(x_ax[badargs], aves[x][badargs], yerr=aves[x + 40][badargs], fmt='or', ms=0,
                          mfc='None', capsize=1.5, elinewidth=1, label='bad runs', alpha=0.5)
        ax[i, j].axhline(tot_ave[x - 3], 0, 1, c='orange', ls="--", label=r"$\mu_w$")
        ax[i, j].axhline(high_point[x - 3], 0, 1, c='blue', ls="--", label=r'4$\sigma_w$')
        ax[i, j].axhline(low_point[x - 3], 0, 1, c='blue', ls="--")
        ax[i, j].set_ylabel(y_labels[x - 3], fontsize=10)
        ax[i, j].set_xlabel("Run Index", fontsize=10)
        # ax[i, j].legend(fontsize=8)
ax[-1, -1].set_axis_off()
ax[-1, -1].errorbar(1, 1, yerr=1, c='k', ms=0, mfc='None', lw=0,
                    capsize=2.5, elinewidth=2, label=r'good runs')
ax[-1, -1].errorbar(1, 1, yerr=1, c='r', ms=0, mfc='None', lw=0,
                    capsize=2.5, elinewidth=2, label=r'bad runs')
ax[-1, -1].errorbar(1, 1, yerr=1.5, c='white', ms=0, mfc='None', lw=0,
                    capsize=3.5, elinewidth=7, mec='white')
ax[-1, -1].plot(1, c='orange', lw=3, ls='--', label=r"$\mu_w$")
ax[-1, -1].plot(1, c='blue', lw=3, ls='--', label=r'4$\sigma_w$')
ax[-1, -1].legend(fontsize=15, loc='center')
plt.show()

y_labels = []
for i in range(16):
    y_labels.append("EPD Ring {}".format(i+1))
fig, ax = plt.subplots(4, 4, figsize=(12, 7), constrained_layout=True)
for i in range(4):
    for j in range(4):
        x = i * 4 + j + 11
        ax[i, j].errorbar(x_ax, aves[x], yerr=aves[x + 40], fmt='ok', ms=0,
                          mfc='None', capsize=1.5, elinewidth=1)
        ax[i, j].errorbar(x_ax[badargs], aves[x][badargs], yerr=aves[x + 40][badargs], fmt='or', ms=0,
                          mfc='None', capsize=1.5, elinewidth=1, alpha=0.5)
        ax[i, j].axhline(tot_ave[x - 3], 0, 1, c='orange', ls="--", label=r"$\mu_w$")
        ax[i, j].axhline(high_point[x - 3], 0, 1, c='blue', ls="--", label=r'4$\sigma_w$')
        ax[i, j].axhline(low_point[x - 3], 0, 1, c='blue', ls="--")
        ax[i, j].set_ylabel(y_labels[x - 11], fontsize=10)
        ax[i, j].set_xlabel("Run Index", fontsize=10)
        ax[i, j].legend(fontsize=8, loc='lower right')
plt.show()

y_labels = []
for i in range(16):
    y_labels.append("EPD Ring {}".format(i+17))
fig, ax = plt.subplots(4, 4, figsize=(12, 7), constrained_layout=True)
for i in range(4):
    for j in range(4):
        x = i * 4 + j + 27
        ax[i, j].errorbar(x_ax, aves[x], yerr=aves[x + 40], fmt='ok', ms=0,
                          mfc='None', capsize=1.5, elinewidth=1)
        ax[i, j].errorbar(x_ax[badargs], aves[x][badargs], yerr=aves[x + 40][badargs], fmt='or', ms=0,
                          mfc='None', capsize=1.5, elinewidth=1, alpha=0.5)
        ax[i, j].axhline(tot_ave[x - 3], 0, 1, c='orange', ls="--", label=r"$\mu_w$")
        ax[i, j].axhline(high_point[x - 3], 0, 1, c='blue', ls="--", label=r'4$\sigma_w$')
        ax[i, j].axhline(low_point[x - 3], 0, 1, c='blue', ls="--")
        ax[i, j].set_ylabel(y_labels[x - 27], fontsize=10)
        ax[i, j].set_xlabel("Run Index", fontsize=10)
        ax[i, j].legend(fontsize=8, loc='lower right')
plt.show()

badruns = np.unique(np.hstack((badruns, emptyruns)))
# np.save(r'D:\14GeV\Thesis\goodruns.npy', goodruns)
np.save(r'D:\14GeV\Thesis\more_badruns.npy', badruns)
