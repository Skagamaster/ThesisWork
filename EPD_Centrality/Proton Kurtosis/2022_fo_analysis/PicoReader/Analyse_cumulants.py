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

# Let's make some plots of the error bars using delta for the ratio and min/max, to see how they compare.
# This code is going to be ugly because I just need it to work. Deal with it.
c_r_comp = np.zeros((3, 4, 11))
e_r_comp_up = np.zeros((3, 4, 11))
e_r_comp_down = np.zeros((3, 4, 11))
for i in range(3):
    c_r_comp[i][0] = c[i][0]
    c_r_comp[i][1] = np.divide(c[i][1], c[i][0])
    c_r_comp[i][2] = np.divide(c[i][2], c[i][1])
    c_r_comp[i][3] = np.divide(c[i][3], c[i][1])
    e_r_comp_up[i][0] = c[i][0] + e[i][0]
    e_r_comp_up[i][1] = np.divide(c[i][1]+e[i][1], c[i][0]-e[i][0])
    e_r_comp_up[i][2] = np.divide(c[i][2]+e[i][2], c[i][1]-e[i][1])
    e_r_comp_up[i][3] = np.divide(c[i][3]+e[i][3], c[i][1]-e[i][1])
    e_r_comp_down[i][0] = c[i][0] - e[i][0]
    e_r_comp_down[i][1] = np.divide(c[i][1]-e[i][1], c[i][0]+e[i][0])
    e_r_comp_down[i][2] = np.divide(c[i][2]-e[i][2], c[i][1]+e[i][1])
    e_r_comp_down[i][3] = np.divide(c[i][3]-e[i][3], c[i][1]+e[i][1])

x_cent = ['90-100%', '80-90%', '70-80%', '60-70%', '50-60%',
          '40-50%', '30-40%', '20-30%', '10-20%', '5-10%', '0-5%']
x = np.linspace(0, 10, 11)
x_offset = np.linspace(0.3, 10.3, 11)
ms = 10
ew = 1
c1 = 'r'
c2 = 'k'
c3 = 'purple'
c_arr = [c1, c2, c3]

plt.fill_between(x, c_r[0][0]-e_r[0][0], c_r[0][0]+e_r[0][0], color='r')
plt.errorbar(x, c_r[0][0], yerr=e_r[0][0], marker='s', color='k', ms=ms,
             mfc='None', elinewidth=ew, lw=0, ecolor='k', label=r'$X_{RM3}$, Delta')
# plt.show()
plt.close()

plt.fill_between(x, c_r[0][1]-e_r[0][1], c_r[0][1]+e_r[0][1], color='r')
plt.errorbar(x, c_r[0][1], yerr=e_r[0][1], marker='s', color='k', ms=ms,
             mfc='None', elinewidth=ew, lw=0, ecolor='k', label=r'$X_{RM3}$, Delta')
# plt.show()
plt.close()

labels = [r'$C_1$', r'$C_2$', r'$C_3$', r'$C_4$']
fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for i in range(2):
    for j in range(2):
        r = i*2 + j
        ax[i, j].errorbar(x[3:], c[2][r][3:], yerr=e[2][r][3:], marker='s', color=c1, ms=ms,
                          mfc='None', elinewidth=ew, lw=0, ecolor=c1, label=r'$X_{ReLU}$: Delta')
        ax[i, j].errorbar(x[3:], c[0][r][3:], yerr=e[0][r][3:], marker='*', color=c2, ms=ms,
                          mfc='None', elinewidth=ew, lw=0, ecolor=c2, label=r'$X_{RM3}$: Delta')
        ax[i, j].legend()
        ax[i, j].set_ylabel(labels[r], fontsize=20)
        ax[i, j].set_xlabel("Centrality", fontsize=20)
plt.show()
plt.close()

labels = [r'$C_1$', r'$\frac{C_2}{C_1}$', r'$\frac{C_3}{C_2}$', r'$\frac{C_4}{C_2}$']
fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for i in range(2):
    for j in range(2):
        r = i*2 + j
        ax[i, j].fill_between(x[3:], e_r_comp_down[0][r][3:], e_r_comp_up[0][r][3:], color=c2, alpha=0.5,
                              label=r'$X_{RM3}$: min/max')
        ax[i, j].fill_between(x[3:], e_r_comp_down[2][r][3:], e_r_comp_up[2][r][3:], color=c1, alpha=0.5,
                              label=r'$X_{ReLU}$: min/max')
        #ax[i, j].errorbar(x[3:], c_r[2][r][3:], yerr=e_r[2][r][3:], marker='s', color=c1, ms=ms,
        #                  mfc='None', elinewidth=ew, lw=0, ecolor=c1, label=r'$X_{ReLU}$: Delta')
        ax[i, j].legend()
        ax[i, j].set_ylabel(labels[r], fontsize=20)
        ax[i, j].set_xlabel("Centrality", fontsize=20)
plt.show()
plt.close()

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
plt.show()
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
# plt.show()
plt.close()

"""
Values for $X_{RM3}$:
['90-100%', '80-90%', '70-80%', '60-70%', '50-60%', '40-50%', '30-40%', '20-30%', '10-20%', '5-10%', '0-5%']
[5.06889145 2.53585454 2.11968858 1.97223462 1.94609139 1.56975081
 1.37757687 1.27671902 1.18831657 1.15073472 1.10837221]
[3.63899553 1.94443328 1.63952187 1.5246166  1.52235131 1.33467769
 1.21936737 1.14812354 1.08630619 1.056879   1.00779884]

Values for $X_{ReLU}$:
['90-100%', '80-90%', '70-80%', '60-70%', '50-60%', '40-50%', '30-40%', '20-30%', '10-20%', '5-10%', '0-5%']
[4.54519419 2.01527269 2.13922451 1.91214047 1.83341753 1.75654046
 1.63820898 1.54372739 1.43256985 1.36895263 1.24891252]
[3.88190559 1.68161856 1.69723079 1.62140682 1.56703928 1.54490191
 1.46269034 1.3914958  1.30892733 1.25536368 1.15340497]
"""
x_rm3_up = [5.06889145, 2.53585454, 2.11968858, 1.97223462, 1.94609139, 1.56975081,
            1.37757687, 1.27671902, 1.18831657, 1.15073472, 1.10837221]
x_rm3_down = [3.63899553, 1.94443328, 1.63952187, 1.5246166,  1.52235131, 1.33467769,
              1.21936737, 1.14812354, 1.08630619, 1.056879,   1.00779884]
x_relu_up = [4.54519419, 2.01527269, 2.13922451, 1.91214047, 1.83341753, 1.75654046,
             1.63820898, 1.54372739, 1.43256985, 1.36895263, 1.24891252]
x_relu_down = [3.88190559, 1.68161856, 1.69723079, 1.62140682, 1.56703928, 1.54490191,
               1.46269034, 1.3914958,  1.30892733, 1.25536368, 1.15340497]

ms = 10
start = 5
plt.figure(figsize=(12, 7), constrained_layout=True)
plt.fill_between(x[start:], x_rm3_down[start:], x_rm3_up[start:], color=c1, alpha=0.2,
                 label=r'$X_{RM3}$ UrQMD')
plt.fill_between(x[start:], x_relu_down[start:], x_relu_up[start:], color=c3, alpha=0.2,
                 label=r'$X_{ReLU}$ UrQMD')
'''
plt.errorbar(x[start:], c_r[0][1][start:], yerr=e_r[0][1][start:],
             lw=0, markersize=ms, elinewidth=1, capsize=2,
             label=r'$X_{RM3}$', marker='o', color=c1)
plt.errorbar(x[start:], c_r[2][1][start:], yerr=e_r[2][1][start:],
             lw=0, markersize=ms, elinewidth=1, capsize=2,
             label=r'$X_{ReLU}$', marker='X', color=c3)
'''
plt.errorbar(x[start:], c_r[0][1][start:], yerr=(c_r[0][1][start:]-e_r_comp_down[0][1][start:],
                                                 e_r_comp_up[0][1][start:]-c_r[0][1][start:]),
             lw=0, markersize=ms, elinewidth=1, capsize=2,
             label=r'$X_{RM3}$', marker='o', color=c1)
plt.errorbar(x[start:], c_r[2][1][start:], yerr=(c_r[2][1][start:]-e_r_comp_down[2][1][start:],
                                                 e_r_comp_up[2][1][start:]-c_r[2][1][start:]),
             lw=0, markersize=ms, elinewidth=1, capsize=2,
             label=r'$X_{ReLU}$', marker='X', color=c3)
plt.ylabel(r"$\frac{\sigma^2}{\mu}$", fontsize=20)
plt.xlabel("Centrality", fontsize=15)
plt.xticks(x[start:], x_cent[start:], rotation=45)
plt.legend(fontsize=15)
plt.show()
