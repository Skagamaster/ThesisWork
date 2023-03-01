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

labels = [r'$C_1$', r'$C_2$', r'$C_3$', r'$C_4$']
fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for i in range(2):
    for j in range(2):
        r = i * 2 + j
        ax[i, j].plot(df_relu['e{}'.format(r + 1)])
        ax[i, j].set_ylabel(labels[r], fontsize=20)
        ax[i, j].set_xlabel("Centrality", fontsize=20)
# plt.show()
plt.close()

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
n_arr = np.zeros((3, 4, 11))
for i in range(3):
    index = df[i].index <= cent[i][0]
    for j in range(4):
        arr = df[i]['c{}'.format(j + 1)][index] * df[i]['n'][index]
        arr1 = df[i]['n'][index]
        n_arr[i][j][0] = np.sum(arr1)
        c[i][j][0] = np.sum(arr) / np.sum(arr1)
        arr = df[i]['e{}'.format(j + 1)][index] * df[i]['n'][index]
        checker = len(arr[arr == 0])
        if checker > 0:
            print("There are some zeros at i=", i, "j=", j)
            plt.plot(arr)
            plt.show()
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
            n_arr[i][j][k] = np.sum(df[i]['n'][index])
            arr = df[i]['e{}'.format(j + 1)][index] * df[i]['n'][index]
            checker = len(arr[arr == 0])
            if checker > 0:
                print("There are some zeros at i=", i, "k=", k, "j=", j)
                plt.plot(arr)
                plt.show()
            e[i][j][k] = np.sum(arr) / np.sum(df[i]['n'][index])
            arr = df[i]['cr{}'.format(j + 1)][index] * df[i]['n'][index]
            c_r[i][j][k] = np.sum(arr) / np.sum(df[i]['n'][index])
            arr = df[i]['er{}'.format(j + 1)][index] * df[i]['n'][index]
            e_r[i][j][k] = np.sum(arr) / np.sum(df[i]['n'][index])
    index = cent[i][-1] < df[i].index
    for j in range(4):
        arr = df[i]['c{}'.format(j + 1)][index] * df[i]['n'][index]
        arr = arr[arr != 0]
        c[i][j][-1] = np.sum(arr) / np.sum(df[i]['n'][index])
        n_arr[i][j][-1] = np.sum(df[i]['n'][index])
        arr = df[i]['e{}'.format(j + 1)][index] * df[i]['n'][index]
        arr = arr[arr != 0]
        checker = len(arr[arr == 0])
        e[i][j][-1] = np.sum(arr) / np.sum(df[i]['n'][index])
        if j == 3:
            print("i=", i, "(second) j=", j, "errors=", e[i][j])
        arr = df[i]['cr{}'.format(j + 1)][index] * df[i]['n'][index]
        arr = arr[arr != 0]
        c_r[i][j][-1] = np.sum(arr) / np.sum(df[i]['n'][index])
        arr = df[i]['er{}'.format(j + 1)][index] * df[i]['n'][index]
        arr = arr[arr != 0]
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
    e_r_comp_up[i][1] = np.divide(c[i][1] + e[i][1], c[i][0] - e[i][0])
    e_r_comp_up[i][2] = np.divide(c[i][2] + e[i][2], c[i][1] - e[i][1])
    e_r_comp_up[i][3] = np.divide(c[i][3] + e[i][3], c[i][1] - e[i][1])
    e_r_comp_down[i][0] = c[i][0] - e[i][0]
    e_r_comp_down[i][1] = np.divide(c[i][1] - e[i][1], c[i][0] + e[i][0])
    e_r_comp_down[i][2] = np.divide(c[i][2] - e[i][2], c[i][1] + e[i][1])
    e_r_comp_down[i][3] = np.divide(c[i][3] - e[i][3], c[i][1] + e[i][1])

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

plt.fill_between(x, c_r[0][0] - e_r[0][0], c_r[0][0] + e_r[0][0], color='r')
plt.errorbar(x, c_r[0][0], yerr=e_r[0][0], marker='s', color='k', ms=ms,
             mfc='None', elinewidth=ew, lw=0, ecolor='k', label=r'$X_{RM3}$, Delta')
plt.legend()
# plt.show()
plt.close()

plt.fill_between(x, c_r[0][1] - e_r[0][1], c_r[0][1] + e_r[0][1], color='r')
plt.errorbar(x, c_r[0][1], yerr=e_r[0][1], marker='s', color='k', ms=ms,
             mfc='None', elinewidth=ew, lw=0, ecolor='k', label=r'$X_{RM3}$, Delta')
plt.legend()
# plt.show()
plt.close()

labels = [r'$C_1$', r'$C_2$', r'$C_3$', r'$C_4$']
fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for i in range(2):
    for j in range(2):
        r = i * 2 + j
        if r == 3:
            ax[i, j].set_axis_off()
        else:
            ax[i, j].errorbar(x[3:], c[2][r][3:], yerr=e[2][r][3:], marker='o', color='red', ms=ms,
                              mfc='None', elinewidth=ew, lw=0, ecolor='red', label=r'$X_{ReLU}$')
            ax[i, j].errorbar(x[3:], c[0][r][3:], yerr=e[0][r][3:], marker='X', color='purple', ms=ms,
                              mfc='None', elinewidth=ew, lw=0, ecolor='purple', label=r'$X_{RM3}$')
            ax[i, j].legend()
            ax[i, j].set_ylabel(labels[r], fontsize=20)
            ax[i, j].set_xlabel("Centrality", fontsize=20)
            print(labels[r])
            print('X_{RM3}')
            print(c[2][r][3:])
            print(e[2][r][3:])
            print('X_{ReLU}')
            print(c[0][r][3:])
            print(e[0][r][3:])
plt.show()
plt.close()

labels = [r'$C_1$', r'$C_2$', r'$C_3$', r'$C_4$']
fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for i in range(2):
    for j in range(2):
        r = i * 2 + j
        if r == 3:
            ax[i, j].set_axis_off()
        else:
            ax[i, j].plot(x[3:], n_arr[2][r][3:])
            ax[i, j].set_ylabel("N for " + labels[r], fontsize=20)
            ax[i, j].set_xlabel("Centrality", fontsize=20)
plt.show()
plt.close()

labels = [r'$C_1$', r'$\frac{C_2}{C_1}$', r'$\frac{C_3}{C_2}$', r'$\frac{C_4}{C_2}$']
fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for i in range(2):
    for j in range(2):
        r = i * 2 + j
        if r == 3:
            ax[i, j].set_axis_off()
        else:
            ax[i, j].fill_between(x[3:], e_r_comp_down[0][r][3:], e_r_comp_up[0][r][3:], color='red', alpha=0.5,
                                  label=r'$X_{RM3}$')
            ax[i, j].fill_between(x[3:], e_r_comp_down[2][r][3:], e_r_comp_up[2][r][3:], color='purple', alpha=0.5,
                                  label=r'$X_{ReLU}$')
            ax[i, j].plot(x[3:], c_r[2][r][3:], marker='X', color='purple', ms=10, lw=0)
            ax[i, j].plot(x[3:], c_r[0][r][3:], marker='o', color='red', ms=10, lw=0)
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
x_rm3_down = [3.63899553, 1.94443328, 1.63952187, 1.5246166, 1.52235131, 1.33467769,
              1.21936737, 1.14812354, 1.08630619, 1.056879, 1.00779884]
x_relu_up = [4.54519419, 2.01527269, 2.13922451, 1.91214047, 1.83341753, 1.75654046,
             1.63820898, 1.54372739, 1.43256985, 1.36895263, 1.24891252]
x_relu_down = [3.88190559, 1.68161856, 1.69723079, 1.62140682, 1.56703928, 1.54490191,
               1.46269034, 1.3914958, 1.30892733, 1.25536368, 1.15340497]
x_rm3_mu = [0.16606767, 0.26224656, 0.56895051, 1.18465578, 2.27529973, 3.97187078,
            6.59908649, 10.45517866, 16.46908131, 26.2135397, 28.77554032]
x_rm3_mu_e = [0.00825203, 0.00649494, 0.01168939, 0.02052312, 0.03480135, 0.0500206,
              0.07176641, 0.1013158, 0.12835062, 0.17205322, 0.18788854]
x_relu_mu = [0.21392126, 0.29841374, 0.59679193, 1.2202622, 2.31335071, 4.06501825,
             6.63011771, 10.38519215, 16.28663719, 25.82902518, 28.13083727]
x_relu_mu_e = [0.00757969, 0.00650136, 0.01207012, 0.02082861, 0.03452776, 0.05463543,
               0.07910259, 0.11147203, 0.14148718, 0.16259699, 0.163459]
x_rm3_sigrat = [4.35394349, 2.24014391, 1.87960522, 1.74842561, 1.73422135, 1.45221425,
                1.29847212, 1.21242128, 1.13731138, 1.07619274, 1.05812878]
x_rm3_down = [3.31043575, 1.86830488, 1.59122478, 1.48580416, 1.48664247, 1.31214251,
              1.20315055, 1.13557418, 1.0771792, 1.02044762, 1.00137607]
x_rm3_up = [5.50611682, 2.63219243, 2.18061506, 2.02084026, 1.98983174, 1.59597223,
            1.39591347, 1.29079455, 1.19841011, 1.13274028, 1.11573383]
x_relu_sigrat = [4.21354989, 1.84844562, 1.91822765, 1.76677364, 1.70022841, 1.65072119,
                 1.55044966, 1.46761159, 1.37074859, 1.24118081, 1.20113306]
x_relu_down = [3.60101228, 1.63463669, 1.64886665, 1.58373729, 1.53522716, 1.51757717,
               1.44088796, 1.37386118, 1.29630863, 1.18224928, 1.14658426]
x_relu_up = [4.89924983, 2.07210601, 2.19911424, 1.95641361, 1.87042173, 1.7875759,
             1.66268471, 1.56341369, 1.44652203, 1.30091533, 1.25637478]
x_b_sigrat = [1.10440298, 1.173378, 1.24074777, 1.32928551, 1.39067977, 1.45694794,
              1.51001583, 1.58779016, 1.70043569, 1.77703492, 1.86819857]
x_b_down = [1.04529089, 1.1242906, 1.18723836, 1.26886333, 1.32957385, 1.38699429,
            1.43224067, 1.49317919, 1.54957251, 1.50351754, 1.47991463]
x_b_up = [1.16423966, 1.22303301, 1.29499843, 1.39076552, 1.45305253, 1.5287155,
          1.59039921, 1.68674887, 1.86151323, 2.10014596, 2.34488487]

ms = 10
start = 5
plt.figure(figsize=(12, 7), constrained_layout=True)
plt.fill_between(x[start:], x_rm3_down[start:], x_rm3_up[start:], color=c1, alpha=0.2,
                 label=r'$X_{RM3}$ UrQMD')
plt.fill_between(x[start:], x_relu_down[start:], x_relu_up[start:], color=c3, alpha=0.2,
                 label=r'$X_{ReLU}$ UrQMD')
plt.fill_between(x[start:], x_b_down[::-1][start:], x_b_up[::-1][start:], color='brown', alpha=0.2,
                 label=r'b UrQMD')
plt.errorbar(x[start:], c_r[0][1][start:], yerr=(c_r[0][1][start:] - e_r_comp_down[0][1][start:],
                                                 e_r_comp_up[0][1][start:] - c_r[0][1][start:]),
             lw=0, markersize=ms, elinewidth=1, capsize=2,
             label=r'$X_{RM3}$', marker='o', color=c1)
plt.errorbar(x[start:], c_r[2][1][start:], yerr=(c_r[2][1][start:] - e_r_comp_down[2][1][start:],
                                                 e_r_comp_up[2][1][start:] - c_r[2][1][start:]),
             lw=0, markersize=ms, elinewidth=1, capsize=2,
             label=r'$X_{ReLU}$', marker='X', color=c3)
plt.ylabel(r"$\frac{\sigma^2}{\mu}$", fontsize=20)
plt.xlabel("Centrality", fontsize=15)
plt.xticks(x[start:], x_cent[start:], rotation=45)
plt.legend(fontsize=15)
plt.show()

x_rm3_srat = [13.94497004, 7.02783267, 6.09976256, 6.42897825, 7.37120583, 3.92569748,
              2.47265945, 1.62470803, 1.03105712, 0.76843591, 0.72128479]
x_rm3_down = [5.80643719, 2.87205834, 1.67649985, 1.78295435, 2.94673707, 1.55702336,
              1.05873877, 0.71926845, 0.54664915, 0.32712231, 0.25951943]
x_rm3_up = [27.26718875, 12.81091643, 12.58688393, 12.96611355, 13.30495801, 6.85223332,
            4.13827558, 2.6743745, 1.56671206, 1.25678393, 1.23664081]
b_srat = [0.09499582, 0.79172371, 1.0893306, 1.20804944, 1.17735855, 1.29919072,
          1.21892191, 1.19978325, 1.58763339, 1.35198504, 1.58374667]
x_b_down = [-0.55633197, 0.40387129, 0.67722362, 0.78419029, 0.85449796, 0.90914865,
            0.90076527, 0.8923316, 0.87543171, 0.54658542, 0.45513327]
x_b_up = [0.81828325, 1.20959761, 1.53563749, 1.67275891, 1.52648442, 1.74112425,
          1.56882241, 1.54935717, 2.58051564, 2.96230262, 4.22378993]
x_relu_srat = [7.28826912, 3.73546823, 5.41111444, 4.04382948, 4.12322401, 3.45700264,
               2.75719308, 2.12891225, 1.40158799, 0.94781623, 0.81441334]
x_relu_down = [4.26250993, 1.2150387, 1.56313265, 1.60577019, 1.73733413, 1.74048631,
               1.42931349, 1.1736384, 0.84254774, 0.51439385, 0.4027893]
x_relu_up = [12.89569023, 7.07750801, 10.57621632, 7.12760442, 7.10779858, 5.50712476,
             4.30140713, 3.21582019, 2.01835133, 1.42090531, 1.2633269]

ms = 10
start = 3
plt.figure(figsize=(12, 7), constrained_layout=True)
#plt.fill_between(x[start:], x_rm3_down[start:], x_rm3_up[start:], color=c1, alpha=0.2,
#                 label=r'$X_{RM3}$ UrQMD')
#plt.fill_between(x[start:], x_relu_down[start:], x_relu_up[start:], color=c3, alpha=0.2,
#                 label=r'$X_{ReLU}$ UrQMD')
#plt.fill_between(x[start:], x_b_down[::-1][start:], x_b_up[::-1][start:], color='brown', alpha=0.2,
#                 label=r'b UrQMD')
plt.errorbar(x[start:], c_r[0][2][start:],
             yerr=(c_r[0][2][start:] - e_r_comp_down[0][2][start:], e_r_comp_up[0][2][start:] - c_r[0][2][start:]),
             lw=0, markersize=ms, elinewidth=1, capsize=2,
             label=r'$X_{RM3}$', marker='o', color=c1)
plt.errorbar(x[start:], c_r[2][2][start:],
             yerr=(c_r[2][2][start:] - e_r_comp_down[2][2][start:], e_r_comp_up[2][2][start:] - c_r[2][2][start:]),
             lw=0, markersize=ms, elinewidth=1, capsize=2,
             label=r'$X_{ReLU}$', marker='X', color=c3)
plt.ylabel(r"$S\sigma$", fontsize=20)
plt.xlabel("Centrality", fontsize=15)
plt.xticks(x[start:], x_cent[start:], rotation=45)
plt.legend(fontsize=15)
plt.show()
