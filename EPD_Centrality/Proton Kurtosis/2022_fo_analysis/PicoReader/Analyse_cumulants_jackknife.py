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
from matplotlib.colors import LogNorm
import os
from matplotlib.colors import LogNorm
from scipy.stats import skew, kurtosis, moment
import pico_reader as pr

iterations = 8  # MUST match with "iterations" in Proton_Analysis_jackknife.py!

# Import our cumulant dataframes.
"""
This creates an array which is "iterations" in length, each member having 4 components:
1. RM3 df
2. EPD_Sum df
3. EPD_linear_fit df
4. EPD_ReLU_ML df
Each dataframe has the centrality metric, cumulants c1-c4, cumulant ratios cr1-cr4 (where
cr1 = c1), and number of entries n at that centrality metric.
The "n" values are for doing CBWC only; they don't factor into the jackknife.
"""
dfs = []
for i in range(1, iterations + 1):
    dfs.append([])
    df_rm3 = pd.read_pickle(r'D:\14GeV\Thesis\Proton_Analysis_WIP\ML_Jackknife\rm3_cumulants{}.pkl'.format(i))
    df_sum = pd.read_pickle(r'D:\14GeV\Thesis\Proton_Analysis_WIP\ML_Jackknife\sum_cumulants{}.pkl'.format(i))
    df_lin = pd.read_pickle(r'D:\14GeV\Thesis\Proton_Analysis_WIP\ML_Jackknife\lin_cumulants{}.pkl'.format(i))
    df_relu = pd.read_pickle(r'D:\14GeV\Thesis\Proton_Analysis_WIP\ML_Jackknife\relu_cumulants{}.pkl'.format(i))
    dfs[i - 1].append(df_rm3)
    dfs[i - 1].append(df_sum)
    dfs[i - 1].append(df_lin)
    dfs[i - 1].append(df_relu)

labels = [r'$C_1$', r'$C_2$', r'$C_3$', r'$C_4$']

# Find percentile (quantile) based centrality.
cent_range = (95, 90, 80, 70, 60, 50, 40, 30, 20, 10)
cent = np.zeros((4, 10))
rm_class = ['RefMult3', 'epd_sum', 'epd_lin', 'epd_relu']
rm_vals = []
for i in range(4):  # For the RM class.
    arr = []
    for j in range(iterations):
        arr = np.hstack((arr, np.repeat(dfs[j][i][rm_class[i]].to_numpy(), dfs[j][i]['n'].to_numpy())))
    cent[i] = np.percentile(arr, cent_range)[::-1]
    rm_vals.append(np.unique(arr))

# These are from GMC simulations for RM3 and ReLU (not really possible to do GMC on the others).
cent[0] = [2., 6., 14., 29., 55., 94., 153., 239., 362., 446.]
cent[3] = [2., 7., 17., 33., 60., 100., 157., 238., 351., 428.]

# Get the distributions for the cumulants for each integer RM.
c_arr = [[[], [], [], []],
         [[], [], [], []],
         [[], [], [], []],
         [[], [], [], []]]
c_mv = [[[[], []], [[], []], [[], []], [[], []]],
        [[[], []], [[], []], [[], []], [[], []]],
        [[[], []], [[], []], [[], []], [[], []]],
        [[[], []], [[], []], [[], []], [[], []]]]
n = []
for i in range(4):
    n.append(np.zeros(len(rm_vals[i])))
    for j in range(len(rm_vals[i])):
        for k in range(4):
            c_arr[i][k].append([])

for i in range(4):  # RM type loop
    for j in range(iterations):  # Jackknife iteration loop
        df = dfs[j][i]
        for k in range(len(rm_vals[i])):
            if rm_vals[i][k] in df[rm_class[i]].to_numpy():
                arr = df['n'][df[rm_class[i]] == rm_vals[i][k]]
                if len(arr) > 0:
                    n[i][k] += df['n'][df[rm_class[i]] == rm_vals[i][k]].to_numpy()[0]
            for r in range(4):
                arr = df['c{}'.format(r + 1)][df[rm_class[i]] == rm_vals[i][k]]
                if len(arr) > 0:
                    c_arr[i][r][k].append(arr.to_numpy()[0])

for i in range(4):
    for j in range(len(rm_vals[i])):
        for k in range(4):
            m = np.mean(c_arr[i][k][j])
            v = np.var(c_arr[i][k][j])
            c_mv[i][k][0].append(m)
            c_mv[i][k][1].append(np.sqrt(v / len(c_arr[i][k][j])))

rm_label = [r'$X_{RM3}$', r'$X_{\Sigma}$', r'$X_{LW}$', r'$X_{ReLU}$']
for r in range(4):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for i in range(2):
        for j in range(2):
            x = i * 2 + j
            index = c_mv[x][r][1] > 0
            print(c_mv[x][r][1])
            ax[i, j].errorbar(rm_vals[x][index], c_mv[x][r][0][index], yerr=c_mv[x][r][1][index],
                              marker='o', color='k', ms=2, mfc='None', elinewidth=1, lw=0, ecolor='k')
            ax[i, j].set_xlabel(rm_label[x], fontsize=15)
            ax[i, j].set_ylabel(labels[r], fontsize=15)
    plt.show()
    plt.close()

fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for i in range(2):
    for j in range(2):
        x = i * 2 + j
        ax[i, j].plot(rm_vals[x], n[x])
        ax[i, j].set_xlabel(rm_class[x])
        ax[i, j].set_ylabel('n')
        ax[i, j].set_yscale('log')
plt.show()
plt.close()
