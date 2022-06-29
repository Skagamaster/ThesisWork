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
from scipy.stats import skew, kurtosis, moment
import pico_reader as pr

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

# Now to do a moment analysis on all dfs.
c = [[], [], [], []]
e = [[], [], [], []]
c_r = [[], [], []]
e_r = [[], [], []]
x_vals = []
n_vals = []
for i in range(4):
    for j in range(12):
        c[i].append([])
        e[i].append([])
        if i < 3:
            c_r[i].append([])
            e_r[i].append([])
for i in range(12):
    n_vals.append([])
    if i == 10:  # df full
        vals = np.unique(df['RefMult3'])
    elif i == 11:  # df_trunc
        vals = np.unique(df_trunc['RefMult3'])
    else:
        vals = np.unique(big_df[i]['RefMult3'])
    x_vals.append(vals)
    for m in vals:
        if i == 10:
            arr = df['net_protons'][df['RefMult3'] == m]
        elif i == 11:
            arr = df_trunc['net_protons'][df_trunc['RefMult3'] == m]
        else:
            arr = big_df[i]['net_protons'][big_df[i]['RefMult3'] == m]
        n_vals[i].append(len(arr))
        n = np.sqrt(len(arr))
        u = pr.moment_arr(arr)
        c[0][i].append(u[0])
        c[1][i].append(u[1])
        c[2][i].append(u[2])
        c[3][i].append(u[3] - 3 * (u[1] ** 2))

        err = pr.err(arr, u)
        e[0][i].append(np.sqrt(err[0]) / n)
        e[1][i].append(np.sqrt(err[1]) / n)
        e[2][i].append(np.sqrt(err[2]) / n)
        e[3][i].append(np.sqrt(err[3]) / n)

        c_r[0][i].append(u[1]/np.max((u[0], 1e-6)))
        c_r[1][i].append(u[2]/np.max((u[1], 1e-6)))
        c_r[2][i].append((u[3] - 3 * (u[1] ** 2))/np.max((u[1], 1e-6)))

        err_rat = pr.err_rat(arr, u)
        e_r[0][i].append(err_rat[0] / n)
        e_r[1][i].append(err_rat[1]/n)
        e_r[2][i].append(err_rat[2]/n)
df_c = pd.DataFrame(c[0][11])
df_c.columns = ['c1']
df_c.index = x_vals[11]
for i in range(2, 5):
    df_c['c{}'.format(i)] = c[i-1][11]
for i in range(1, 5):
    df_c['e{}'.format(i)] = e[i-1][11]
for i in range(1, 4):
    df_c['cr{}'.format(i)] = c_r[i-1][11]
    df_c['er{}'.format(i)] = e_r[i - 1][11]
df_c['n'] = n_vals[11]
df_c.to_pickle(r'D:\14GeV\Thesis\PythonArrays\Analysis_Proton_Arrays\trunc_cumulants.pkl')
df_trunc.to_pickle(r'D:\14GeV\Thesis\PythonArrays\Analysis_Proton_Arrays\trunc_set.pkl')
print("All pickled.")

# Now some plots of the cumulants of net-protons by RefMult3 integer, for each df.
t_labels = [r'$\mu$', r'$\sigma^2$', r'$S$', r'$\kappa$']
ranges = ((0, 650), (0, 40), (0, 60), (-100, 100), (-100, 100))
text_high = (35, 55, 95, 95)
# No error
for q in range(4):
    fig, ax = plt.subplots(3, 4, figsize=(16, 9), constrained_layout=True)
    for i in range(3):
        for j in range(4):
            k = i * 4 + j
            ax[i, j].scatter(x_vals[k], c[q][k],
                             marker='o', c='k', s=10)
            ax[i, j].set_xlabel(r"$X_{RM3}$", fontsize=12, loc='right')
            ax[i, j].set_ylabel(t_labels[q], fontsize=15)
            ax[i, j].set_xlim(ranges[0])
            ax[i, j].set_ylim(ranges[q+1])
            if k == 10:
                ax[i, j].text(x=50, y=text_high[q], s="Raw", fontsize=15)
            elif k == 11:
                ax[i, j].text(x=50, y=text_high[q], s="Truncated", fontsize=15)
            else:
                ax[i, j].text(x=50, y=text_high[q], s=labels[k], fontsize=15)
    plt.show()
    plt.close()
# With error
for q in range(4):
    fig, ax = plt.subplots(3, 4, figsize=(16, 9), constrained_layout=True)
    for i in range(3):
        for j in range(4):
            k = i * 4 + j
            ax[i, j].errorbar(x_vals[k], c[q][k], yerr=e[q][k],
                              marker='o', color='k', ms=2,
                              mfc='None', elinewidth=0.1, lw=0,
                              ecolor='black')
            ax[i, j].set_xlabel(r"$X_{RM3}$", fontsize=12, loc='right')
            ax[i, j].set_ylabel(t_labels[q], fontsize=15)
            ax[i, j].set_xlim(ranges[0])
            ax[i, j].set_ylim(ranges[q + 1])
            if k == 10:
                ax[i, j].text(x=50, y=text_high[q], s="Raw", fontsize=15)
            elif k == 11:
                ax[i, j].text(x=50, y=text_high[q], s="Truncated", fontsize=15)
            else:
                ax[i, j].text(x=50, y=text_high[q], s=labels[k], fontsize=15)
    plt.show()
    plt.close()

# Now again, but for the ratios.
t_labels = [r'$\frac{\sigma^2}{\mu}$', r'$S\sigma$', r'$\kappa\sigma^2$']
ranges = ((0, 650), (0.5, 3.5), (-100, 100), (-1000, 1000))
text_high = (3.0, 95, 95)
# No error
for q in range(3):
    fig, ax = plt.subplots(3, 4, figsize=(16, 9), constrained_layout=True)
    for i in range(3):
        for j in range(4):
            k = i * 4 + j
            ax[i, j].scatter(x_vals[k], c_r[q][k], marker='o', c='k', s=10)
            ax[i, j].set_xlabel(r"$X_{RM3}$", fontsize=12, loc='right')
            ax[i, j].set_ylabel(t_labels[q], fontsize=15)
            ax[i, j].set_xlim(ranges[0])
            ax[i, j].set_ylim(ranges[q + 1])
            if k == 10:
                ax[i, j].text(x=50, y=text_high[q], s="Raw", fontsize=15)
            elif k == 11:
                ax[i, j].text(x=50, y=text_high[q], s="Truncated", fontsize=15)
            else:
                ax[i, j].text(x=50, y=text_high[q], s=labels[k], fontsize=15)
    plt.show()
    plt.close()
# With error
for q in range(3):
    fig, ax = plt.subplots(3, 4, figsize=(16, 9), constrained_layout=True)
    for i in range(3):
        for j in range(4):
            k = i * 4 + j
            ax[i, j].errorbar(x_vals[k], c_r[q][k], yerr=e_r[q][k],
                              marker='o', color='k', ms=2,
                              mfc='None', elinewidth=0.1, lw=0,
                              ecolor='black')
            ax[i, j].set_xlabel(r"$X_{RM3}$", fontsize=12, loc='right')
            ax[i, j].set_ylabel(t_labels[q], fontsize=15)
            ax[i, j].set_xlim(ranges[0])
            ax[i, j].set_ylim(ranges[q + 1])
            if k == 10:
                ax[i, j].text(x=50, y=text_high[q], s="Raw", fontsize=15)
            elif k == 11:
                ax[i, j].text(x=50, y=text_high[q], s="Truncated", fontsize=15)
            else:
                ax[i, j].text(x=50, y=text_high[q], s=labels[k], fontsize=15)
    plt.show()
    plt.close()
