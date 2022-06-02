#
# \Read averages for the found protons in good runs and make more bad runs.
#
# \author Skipper Kagamaster
# \date 06/01/2022
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
# /

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir(r"D:\14GeV\Thesis\PythonArrays\Analysis_Histograms")
proton_ave = np.load('proton_ave.npy', allow_pickle=True)
proton_std = np.load('proton_std.npy', allow_pickle=True)
runlist = np.load('runlist.npy', allow_pickle=True)
x = np.linspace(0, len(runlist) - 1, len(runlist))

labels = [[['<protons>', r'0.4 $\leq p_T \leq$ 0.8'],
           ['<antiprotons>', r'0.4 $\leq p_T \leq$ 0.8']],
          [['<protons>', r'0.8 $\leq p_T \leq$ 2.0'],
           ['<antiprotons>', r'0.8 $\leq p_T \leq$ 2.0']]]

fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
aves = np.mean(proton_ave, axis=2)
stds = np.std(proton_ave, axis=2)
p_high = aves + stds
p_low = aves - stds

badruns = []
index = [[[], []], [[], []]]
for i in range(2):
    for j in range(2):
        for k in range(len(runlist)):
            if (proton_ave[i][j][k] < p_low[i][j]) | (proton_ave[i][j][k] > p_high[i][j]):
                badruns.append(runlist[k])
                index[i][j].append(k)
badruns = np.unique(badruns)

for i in range(2):
    for j in range(2):
        ax[i, j].errorbar(x, proton_ave[i][j], yerr=proton_std[i][j],
                          marker='o', color='k', ms=2,
                          mfc='None', elinewidth=0.1, lw=0,
                          label="Good Runs", ecolor='black')
        ax[i, j].errorbar(x[index[i][j]], proton_ave[i][j][index[i][j]],
                          yerr=proton_std[i][j][index[i][j]],
                          marker='o', color='orange', ms=2,
                          mfc='None', elinewidth=0.2, lw=0,
                          label="Bad Runs", ecolor='orange')
        ax[i, j].axhline(aves[i][j], c='r', ls='--', label=r"$\mu$")
        ax[i, j].axhline(aves[i][j] + stds[i][j], c='b', ls='--', label=r"$\sigma$")
        ax[i, j].axhline(aves[i][j] - stds[i][j], c='b', ls='--')
        ax[i, j].set_ylabel(labels[0][j][0], fontsize=15)
        ax[i, j].set_xlabel('Run ID', loc='right', fontsize=15)
        ax[i, j].set_title(labels[i][0][1], fontsize=20)
plt.show()
