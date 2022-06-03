#
# \Analyse proton arrays (but first, find out why these arrays are so janky!).
#
# \author Skipper Kagamaster
# \date 06/02/2022
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
# /

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LogNorm

df = pd.read_pickle(r'D:\14GeV\Thesis\PythonArrays\Analysis_Proton_Arrays\full_set.pkl')
df['net_protons'] = df['protons'] - df['antiprotons']

# Let's do a plot of the problem.
fig, ax = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)
ax[0].hist(df['RefMult3'], bins=650, range=(0, 650), histtype='step', density=True)
ax[0].set_xlabel(r"$X_{RM3}$", fontsize=15, loc='right')
ax[0].set_ylabel(r"$\frac{dN}{d(X_{RM3})}$", fontsize=15, loc='top')
ax[0].set_yscale('log')
ax[1].hist(df['net_protons'][df['RefMult3'] > 472], bins=70, range=(0, 70), histtype='step',
           density=True)
ax[1].set_xlabel(r"$N_p$, 0-5%", fontsize=15, loc='right')
ax[1].set_ylabel(r"$\frac{dN}{dN_p}$", fontsize=15, loc='top')
ax[1].set_yscale('log')
count, binsX, binsY = np.histogram2d(df['net_protons'], df['RefMult3'], bins=(60, 650),
                                     range=((0, 60), (0, 650)))
X, Y = np.meshgrid(binsX, binsY)
im = ax[2].pcolormesh(X, Y, count.T, cmap='jet', norm=LogNorm())
fig.colorbar(im, ax=ax[2])
ax[2].set_xlabel(r"$N_p$", fontsize=15, loc='right')
ax[2].set_ylabel(r"$X_{RM3}$", fontsize=15, loc='top')
# plt.show()
plt.close()

# Now to see where RefMult3 goes off the rails, in increments of 10%.
df_snip = int(len(df) / 10)
df1 = df[:df_snip]
df2 = df[df_snip:2 * df_snip]
df3 = df[2 * df_snip:3 * df_snip]
df4 = df[3 * df_snip:4 * df_snip]
df5 = df[4 * df_snip:5 * df_snip]
df6 = df[5 * df_snip:6 * df_snip]
df7 = df[6 * df_snip:7 * df_snip]
df8 = df[7 * df_snip:8 * df_snip]
df9 = df[8 * df_snip:9 * df_snip]
df10 = df[9 * df_snip:]
big_df = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10]

fig, ax = plt.subplots(3, 4, figsize=(16, 9), constrained_layout=True)
labels = [r"Runs 0-10%", r"Runs 10-20%", r"Runs 20-30%", r"Runs 30-40%", r"Runs 40-50%",
          r"Runs 50-60%", r"Runs 60-70%", r"Runs 70-80%", r"Runs 80-90%", r"Runs 90-100%"]
for i in range(3):
    for j in range(4):
        x = i * 4 + j
        if x > 9:
            ax[i, j].set_axis_off()
            continue
        count, bins = np.histogram(big_df[x]['RefMult3'], bins=650, range=(0, 650))
        ax[i, j].hist(big_df[x]['RefMult3'], bins=650, range=(0, 650), histtype='step')
        ax[i, j].text(x=250, y=0.55 * np.max(count), s=labels[x], fontsize=15)
        ax[i, j].set_xlabel("RefMult3", fontsize=15, loc='right')
        ax[i, j].set_ylabel("Count", fontsize=15, loc='top')
        ax[i, j].set_yscale('log')
# plt.show()
plt.close()

# Truncated dataframe based on the RefMult3 distributions that don't have a strange discontinuity at ~40.
df_trunc = pd.concat([df1, df2, df8, df9, df10], ignore_index=True)
# Let's do a plot of the truncated dataframe.
fig, ax = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)
ax[0].hist(df_trunc['RefMult3'], bins=650, range=(0, 650), histtype='step', density=True)
ax[0].set_xlabel(r"$X_{RM3}$", fontsize=15, loc='right')
ax[0].set_ylabel(r"$\frac{dN}{d(X_{RM3})}$", fontsize=15, loc='top')
ax[0].set_yscale('log')
ax[1].hist(df_trunc['net_protons'][df_trunc['RefMult3'] > 472], bins=70, range=(0, 70), histtype='step',
           density=True)
ax[1].set_xlabel(r"$N_p$, 0-5%", fontsize=15, loc='right')
ax[1].set_ylabel(r"$\frac{dN}{dN_p}$", fontsize=15, loc='top')
ax[1].set_yscale('log')
count, binsX, binsY = np.histogram2d(df_trunc['net_protons'], df_trunc['RefMult3'], bins=(60, 650),
                                     range=((0, 60), (0, 650)))
X, Y = np.meshgrid(binsX, binsY)
im = ax[2].pcolormesh(X, Y, count.T, cmap='jet', norm=LogNorm())
fig.colorbar(im, ax=ax[2])
ax[2].set_xlabel(r"$N_p$", fontsize=15, loc='right')
ax[2].set_ylabel(r"$X_{RM3}$", fontsize=15, loc='top')
# plt.show()
plt.close()

# Now some plots of the average net-protons by RefMult3 integer, for each df.
fig, ax = plt.subplots(3, 4, figsize=(16, 9), constrained_layout=True)
for i in range(3):
    for j in range(4):
        k = i * 4 + j
        if k == 10:
            vals = np.unique(df['RefMult3'])
            aves = []
            stds = []
            for m in vals:
                arr = df['net_protons'][df['RefMult3'] == m]
                aves.append(np.mean(arr))
                stds.append(np.std(arr) / np.sqrt(len(arr)))
            ax[i, j].errorbar(vals, aves, yerr=stds,
                              marker='o', color='k', ms=2,
                              mfc='None', elinewidth=0.1, lw=0,
                              ecolor='black')
            ax[i, j].set_xlabel(r"$X_{RM3}$", fontsize=12, loc='right')
            ax[i, j].set_ylabel(r"$\mu$", fontsize=15)
            ax[i, j].text(x=50, y=35, s="Raw", fontsize=15)
        elif k == 11:
            vals = np.unique(df_trunc['RefMult3'])
            aves = []
            stds = []
            for m in vals:
                arr = df_trunc['net_protons'][df_trunc['RefMult3'] == m]
                aves.append(np.mean(arr))
                stds.append(np.std(arr) / np.sqrt(len(arr)))
            ax[i, j].errorbar(vals, aves, yerr=stds,
                              marker='o', color='k', ms=2,
                              mfc='None', elinewidth=0.1, lw=0,
                              ecolor='black')
            ax[i, j].set_xlabel(r"$X_{RM3}$", fontsize=12, loc='right')
            ax[i, j].set_ylabel(r"$\mu$", fontsize=15)
            ax[i, j].text(x=50, y=35, s="Truncated", fontsize=15)
        else:
            vals = np.unique(big_df[k]['RefMult3'])
            aves = []
            stds = []
            for m in vals:
                arr = big_df[k]['net_protons'][big_df[k]['RefMult3'] == m]
                aves.append(np.mean(arr))
                stds.append(np.std(arr) / np.sqrt(len(arr)))
            ax[i, j].errorbar(vals, aves, yerr=stds,
                              marker='o', color='k', ms=2,
                              mfc='None', elinewidth=0.1, lw=0,
                              ecolor='black')
            ax[i, j].set_xlabel(r"$X_{RM3}$", fontsize=12, loc='right')
            ax[i, j].set_ylabel(r"$\mu$", fontsize=15)
            ax[i, j].text(x=50, y=35, s=labels[k], fontsize=15)
plt.show()
plt.close()
