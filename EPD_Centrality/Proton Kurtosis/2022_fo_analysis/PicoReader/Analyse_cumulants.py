#
# \Analyse cumulants.
#
# \author Skipper Kagamaster
# \date 06/07/2022
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

# Import our cumulant dataframes.
df_rm3 = pd.read_pickle(r'D:\14GeV\Thesis\Proton_Analysis_WIP\ML\rm3_cumulants.pkl')
df_lin = pd.read_pickle(r'D:\14GeV\Thesis\Proton_Analysis_WIP\ML\lin_cumulants.pkl')
df_relu = pd.read_pickle(r'D:\14GeV\Thesis\Proton_Analysis_WIP\ML\relu_cumulants.pkl')
df = [df_rm3, df_lin, df_relu]

# Find percentile (quantile) based centrality.
cent_range = (95, 90, 80, 70, 60, 50, 40, 30, 20, 10)
cent = np.zeros((3, 10))
for i in range(3):
    arr = np.repeat(df[i].index, df[i]['n'])
    cent[i] = np.percentile(arr, cent_range)[::-1]
print(cent[0])
print(cent[2])
# These are from GMC simulations.
cent[0] = [2., 6., 14., 29., 55., 94., 153., 239., 362., 446.]
cent[2] = [2., 7., 17., 33., 60., 100., 157., 238., 351., 428.]

# Now to CBWC the cumulants and errors.
c = np.zeros((3, 4, 11))
e = np.zeros((3, 4, 11))
c_r = np.zeros((3, 4, 11))
e_r = np.zeros((3, 4, 11))
for i in range(3):
    index = df[i].index <= cent[i][0]
    for j in range(4):
        arr = df[i]['c{}'.format(j + 1)][index] * df[i]['n'][index]
        c[i][j][0] = np.sum(arr) / np.sum(df[i]['n'][index])
        arr = df[i]['e{}'.format(j + 1)][index] * df[i]['n'][index]
        e[i][j][0] = np.sum(arr) / np.sum(df[i]['n'][index])
        arr = df[i]['cr{}'.format(j + 1)][index] * df[i]['n'][index]
        c_r[i][j][0] = np.sum(arr) / np.sum(df[i]['n'][index])
        arr = df[i]['er{}'.format(j + 1)][index] * df[i]['n'][index]
        e_r[i][j][0] = np.sum(arr) / np.sum(df[i]['n'][index])
    for k in range(1, 10):
        index = (cent[i][k - 1] < df[i].index) & (df[i].index <= cent[i][k])
        for j in range(4):
            arr = df[i]['c{}'.format(j + 1)][index] * df[i]['n'][index]
            c[i][j][k] = np.sum(arr) / np.sum(df[i]['n'][index])
            arr = df[i]['e{}'.format(j + 1)][index] * df[i]['n'][index]
            e[i][j][k] = np.sum(arr) / np.sum(df[i]['n'][index])
            arr = df[i]['cr{}'.format(j + 1)][index] * df[i]['n'][index]
            c_r[i][j][k] = np.sum(arr) / np.sum(df[i]['n'][index])
            arr = df[i]['er{}'.format(j + 1)][index] * df[i]['n'][index]
            e_r[i][j][k] = np.sum(arr) / np.sum(df[i]['n'][index])
    index = cent[i][-1] < df[i].index
    for j in range(4):
        arr = df[i]['c{}'.format(j + 1)][index] * df[i]['n'][index]
        c[i][j][-1] = np.sum(arr) / np.sum(df[i]['n'][index])
        arr = df[i]['e{}'.format(j + 1)][index] * df[i]['n'][index]
        e[i][j][-1] = np.sum(arr) / np.sum(df[i]['n'][index])
        arr = df[i]['cr{}'.format(j + 1)][index] * df[i]['n'][index]
        c_r[i][j][-1] = np.sum(arr) / np.sum(df[i]['n'][index])
        arr = df[i]['er{}'.format(j + 1)][index] * df[i]['n'][index]
        e_r[i][j][-1] = np.sum(arr) / np.sum(df[i]['n'][index])

# And let's make some plots, shall we?
x_cent = ['90-100%', '80-90%', '70-80%', '60-70%', '50-60%',
          '40-50%', '30-40%', '20-30%', '10-20%', '5-10%', '0-5%']
x = np.linspace(0, 10, 11)
fig, ax = plt.subplots(1, 3, figsize=(12, 7), constrained_layout=True)
ms = 10
ew = 1
c1 = 'r'
c2 = 'k'
c3 = 'purple'
vals = [r'$\mu$', r'$\sigma^2$', r'$\frac{\sigma^2}{\mu}$']
for i in range(3):
    if i < 2:
        ax[i].errorbar(x, c[0][i], yerr=e[0][i],
                       marker='s', color=c1, ms=ms,
                       mfc='None', elinewidth=ew, lw=0,
                       ecolor=c1, label=r'$X_{RM3}$')
        ax[i].errorbar(x, c[2][i], yerr=e[2][i],
                       marker='X', color=c2, ms=ms,
                       mfc='None', elinewidth=ew, lw=0,
                       ecolor=c2, label=r'$X_{ReLU}$')
        ax[i].errorbar(x, c[1][i], yerr=e[1][i],
                       marker='v', color=c3, ms=ms,
                       mfc='None', elinewidth=ew, lw=0,
                       ecolor=c3, label=r'$X_{\Sigma}$')
    else:
        ax[i].errorbar(x, c_r[0][1], yerr=e_r[0][1],
                       marker='s', color=c1, ms=ms,
                       mfc='None', elinewidth=ew, lw=0,
                       ecolor=c1, label=r'$X_{RM3}$')
        ax[i].errorbar(x, c_r[2][1], yerr=e_r[2][1],
                       marker='X', color=c2, ms=ms,
                       mfc='None', elinewidth=ew, lw=0,
                       ecolor=c2, label=r'$X_{ReLU}$')
        ax[i].errorbar(x, c_r[1][1], yerr=e_r[1][1],
                       marker='v', color=c3, ms=ms,
                       mfc='None', elinewidth=ew, lw=0,
                       ecolor=c3, label=r'$X_{\Sigma}$')
        ax[i].set_ylim(0.5, 3.5)
    ax[i].set_ylabel(vals[i], fontsize=15)
    ax[i].set_xlabel('Centrality', fontsize=15)
    ax[i].set_xticks(x)
    ax[i].set_xticklabels(x_cent, rotation=45)
    ax[i].legend()
# plt.show()
plt.close()

c_r_up = c_r + e_r
c_r_down = c_r - e_r
ms = 80

plt.figure(figsize=(12, 7), constrained_layout=True)
plt.fill_between(x[5:], c_r_down[0][1][5:], c_r_up[0][1][5:], color=c1, alpha=0.2)
plt.fill_between(x[5:], c_r_down[2][1][5:], c_r_up[2][1][5:], color=c3, alpha=0.2)
plt.fill_between(x[5:], c_r_down[1][1][5:], c_r_up[1][1][5:], color=c2, alpha=0.2)
plt.scatter(x[5:], c_r[0][1][5:],
             marker='o', color=c1, s=ms,
             label=r'$X_{RM3}$', alpha=0.5)
plt.scatter(x[5:], c_r[2][1][5:],
             marker='X', color=c3, s=ms,
             label=r'$X_{ReLU}$', alpha=0.5)
plt.scatter(x[5:], c_r[1][1][5:],
             marker='s', color=c2, s=ms,
             label=r'$X_{\Sigma}$', alpha=0.5)
plt.ylabel(r"$\frac{\sigma^2}{\mu}$", fontsize=20)
plt.xlabel("Centrality", fontsize=15)
plt.xticks(x[5:], x_cent[5:], rotation=45)
plt.legend(fontsize=15)
plt.show()
