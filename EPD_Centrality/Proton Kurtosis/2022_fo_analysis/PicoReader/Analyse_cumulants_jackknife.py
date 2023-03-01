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
    df_rm3 = pd.read_pickle(r'C:\200\ML_Jackknife\rm3_cumulants{}.pkl'.format(i))
    df_sum = pd.read_pickle(r'C:\200\ML_Jackknife\sum_cumulants{}.pkl'.format(i))
    df_lin = pd.read_pickle(r'C:\200\ML_Jackknife\lin_cumulants{}.pkl'.format(i))
    df_relu = pd.read_pickle(r'C:\200\ML_Jackknife\relu_cumulants{}.pkl'.format(i))
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

# These are from GMC simulations for RM3 and ReLU for 14 GeV (not really possible to do GMC on the others).
# cent[0] = [2., 6., 14., 29., 55., 94., 153., 239., 362., 446.]
# cent[3] = [2., 7., 17., 33., 60., 100., 157., 238., 351., 428.]

# Get the distributions for the cumulants for each integer RM.
c_arr = [[[], [], [], []],
         [[], [], [], []],
         [[], [], [], []],
         [[], [], [], []]]
c_mv = [[[[], []], [[], []], [[], []], [[], []]],
        [[[], []], [[], []], [[], []], [[], []]],
        [[[], []], [[], []], [[], []], [[], []]],
        [[[], []], [[], []], [[], []], [[], []]]]
cr_arr = [[[], [], [], []],
          [[], [], [], []],
          [[], [], [], []],
          [[], [], [], []]]
cr_mv = [[[[], []], [[], []], [[], []], [[], []]],
         [[[], []], [[], []], [[], []], [[], []]],
         [[[], []], [[], []], [[], []], [[], []]],
         [[[], []], [[], []], [[], []], [[], []]]]
n = []
for i in range(4):
    n.append(np.zeros(len(rm_vals[i])))
    for j in range(len(rm_vals[i])):
        for k in range(4):
            c_arr[i][k].append([])
            cr_arr[i][k].append([])

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
                arr = df['cr{}'.format(r + 1)][df[rm_class[i]] == rm_vals[i][k]]
                if len(arr) > 0:
                    cr_arr[i][r][k].append(arr.to_numpy()[0])

for i in range(4):
    for j in range(len(rm_vals[i])):
        for k in range(4):
            m = np.mean(c_arr[i][k][j])
            v = np.var(c_arr[i][k][j])
            c_mv[i][k][0].append(m)
            c_mv[i][k][1].append(np.sqrt(v / len(c_arr[i][k][j])))
            m = np.mean(cr_arr[i][k][j])
            v = np.var(cr_arr[i][k][j])
            cr_mv[i][k][0].append(m)
            cr_mv[i][k][1].append(np.sqrt(v / len(cr_arr[i][k][j])))

rm_label = [r'$X_{RM3}$', r'$X_{\Sigma}$', r'$X_{LW}$', r'$X_{ReLU}$']
for r in range(4):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for i in range(2):
        for j in range(2):
            x = i * 2 + j
            index = np.asarray(c_mv[x][r][1]) > 0
            ax[i, j].errorbar(np.asarray(rm_vals[x])[index],
                              np.asarray(c_mv[x][r][0])[index],
                              yerr=np.asarray(c_mv[x][r][1])[index],
                              marker='o', color='k', ms=2, mfc='None', elinewidth=1, lw=0, ecolor='k')
            ax[i, j].set_xlabel(rm_label[x], fontsize=15)
            ax[i, j].set_ylabel(labels[r], fontsize=15)
    # plt.show()
    plt.close()

cr_labels = [r'$C_1$', r'$\frac{C_2}{C_1}$', r'$\frac{C_3}{C_2}$', r'$\frac{C_4}{C_2}$']
for r in range(4):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for i in range(2):
        for j in range(2):
            x = i * 2 + j
            index = np.asarray(cr_mv[x][r][1]) > 0
            ax[i, j].errorbar(np.asarray(rm_vals[x])[index],
                              np.asarray(cr_mv[x][r][0])[index],
                              yerr=np.asarray(cr_mv[x][r][1])[index],
                              marker='o', color='k', ms=2, mfc='None', elinewidth=1, lw=0, ecolor='k')
            ax[i, j].set_xlabel(rm_label[x], fontsize=15)
            ax[i, j].set_ylabel(cr_labels[r], fontsize=15)
    # plt.show()
    plt.close()

# Now let's do CBWC.
c_cbwc = [[], [], [], []]
e_cbwc = [[], [], [], []]
cr_cbwc = [[], [], [], []]
er_cbwc = [[], [], [], []]


def arr_maker(a, b, c, d):
    arr = np.asarray(a)[d]
    carr = np.asarray(b)[d]
    narr = np.asarray(c)[d]
    index = arr > 0
    carr = np.asarray(carr[index])
    narr = np.asarray(narr[index])
    numer = np.sum(carr * narr)
    denom = np.sum(narr)
    return numer/denom


for i in range(4):
    centrality = cent[i]
    arr_index = np.asarray(rm_vals[i]) <= centrality[0]
    for j in range(4):
        c_cbwc[i].append([])
        e_cbwc[i].append([])
        cr_cbwc[i].append([])
        er_cbwc[i].append([])
        carr = arr_maker(c_mv[i][j][1], c_mv[i][j][0], n[i], arr_index)
        c_cbwc[i][j].append(carr)
        earr = arr_maker(c_mv[i][j][1], c_mv[i][j][1], n[i], arr_index)
        e_cbwc[i][j].append(earr)
        crarr = arr_maker(cr_mv[i][j][1], cr_mv[i][j][0], n[i], arr_index)
        cr_cbwc[i][j].append(crarr)
        erarr = arr_maker(cr_mv[i][j][1], cr_mv[i][j][1], n[i], arr_index)
        er_cbwc[i][j].append(erarr)
    for k in range(len(centrality)-1):
        arr_index = (np.asarray(rm_vals[i]) > centrality[k]) & (np.asarray(rm_vals[i]) <= centrality[k+1])
        for j in range(4):
            carr = arr_maker(c_mv[i][j][1], c_mv[i][j][0], n[i], arr_index)
            c_cbwc[i][j].append(carr)
            earr = arr_maker(c_mv[i][j][1], c_mv[i][j][1], n[i], arr_index)
            e_cbwc[i][j].append(earr)
            crarr = arr_maker(cr_mv[i][j][1], cr_mv[i][j][0], n[i], arr_index)
            cr_cbwc[i][j].append(crarr)
            erarr = arr_maker(cr_mv[i][j][1], cr_mv[i][j][1], n[i], arr_index)
            er_cbwc[i][j].append(erarr)
    arr_index = np.asarray(rm_vals[i]) > centrality[-1]
    for j in range(4):
        carr = arr_maker(c_mv[i][j][1], c_mv[i][j][0], n[i], arr_index)
        c_cbwc[i][j].append(carr)
        earr = arr_maker(c_mv[i][j][1], c_mv[i][j][1], n[i], arr_index)
        e_cbwc[i][j].append(earr)
        crarr = arr_maker(cr_mv[i][j][1], cr_mv[i][j][0], n[i], arr_index)
        cr_cbwc[i][j].append(crarr)
        erarr = arr_maker(cr_mv[i][j][1], cr_mv[i][j][1], n[i], arr_index)
        er_cbwc[i][j].append(erarr)

x_arr = ['90-100%', '80-90%', '70-80%', '60-70%', '50-60%', '40-50%',
         '30-40%', '20-30%', '10-20%', '5-10%', '0-5%']
for r in range(4):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for i in range(2):
        for j in range(2):
            x = i * 2 + j
            ax[i, j].plot(c_cbwc[x][r][2:])
            ax[i, j].errorbar(x_arr[2:], c_cbwc[x][r][2:], yerr=e_cbwc[x][r][2:],
                              marker='o', color='k', ms=2, mfc='None', elinewidth=1, lw=0, ecolor='k')
            ax[i, j].set_xlabel(rm_label[x], fontsize=15)
            ax[i, j].set_ylabel(labels[r], fontsize=15)
    # plt.show()
    plt.close()

x_arr = ['90-100%', '80-90%', '70-80%', '60-70%', '50-60%', '40-50%',
         '30-40%', '20-30%', '10-20%', '5-10%', '0-5%']
for r in range(4):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for i in range(2):
        for j in range(2):
            x = i * 2 + j
            ax[i, j].plot(cr_cbwc[x][r][2:])
            ax[i, j].errorbar(x_arr[2:], cr_cbwc[x][r][2:], yerr=er_cbwc[x][r][2:],
                              marker='o', color='k', ms=2, mfc='None', elinewidth=1, lw=0, ecolor='k')
            ax[i, j].set_xlabel(rm_label[x], fontsize=15)
            ax[i, j].set_ylabel(cr_labels[r], fontsize=15)
    # plt.show()
    plt.close()

fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
x_range = np.linspace(1, len(cr_cbwc[0][0]), len(cr_cbwc[0][0]))
marker = ['o', '*', 'p', 's']
color = ['k', 'orange', 'blue', 'green']
for i in range(2):
    for j in range(2):
        x = i * 2 + j
        for k in range(4):
            ax[i, j].errorbar(x_range[2:]+(0.1*k), cr_cbwc[k][x][2:], yerr=er_cbwc[k][x][2:],
                              marker=marker[k], color=color[k], ms=10, mfc=color[k], elinewidth=1, lw=0,
                              ecolor=color[k], label=rm_label[k], alpha=0.5)
        ax[i, j].set_xlabel('Centrality class', fontsize=15)
        ax[i, j].set_xticks(x_range[2:])
        ax[i, j].set_xticklabels(x_arr[2:], minor=False, rotation=45)
        ax[i, j].set_ylabel(cr_labels[x], fontsize=15)
        ax[i, j].legend()
plt.show()
plt.close()

fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
x_range = np.linspace(1, len(cr_cbwc[0][0]), len(cr_cbwc[0][0]))
marker = ['o', '*', 'p', 's']
color = ['k', 'orange', 'blue', 'green']
for i in range(2):
    for j in range(2):
        x = i * 2 + j
        for k in range(4):
            ax[i, j].errorbar(x_range[2:]+(0.1*k), c_cbwc[k][x][2:], yerr=e_cbwc[k][x][2:],
                              marker=marker[k], color=color[k], ms=10, mfc=color[k], elinewidth=1, lw=0,
                              ecolor=color[k], label=rm_label[k], alpha=0.5)
        ax[i, j].set_xlabel('Centrality class', fontsize=15)
        ax[i, j].set_xticks(x_range[2:])
        ax[i, j].set_xticklabels(x_arr[2:], minor=False, rotation=45)
        ax[i, j].set_ylabel(labels[x], fontsize=15)
        ax[i, j].legend()
plt.show()
plt.close()

# And now with just X_RM3 and X_ReLU
fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
x_range = np.linspace(1, len(cr_cbwc[0][0]), len(cr_cbwc[0][0]))
marker = ['o', '*', 'p', 's']
color = ['k', 'orange', 'blue', 'green']
for i in range(2):
    for j in range(2):
        x = i * 2 + j
        for k in (0, 3):
            ax[i, j].errorbar(x_range[2:]+(0.1*k), cr_cbwc[k][x][2:], yerr=er_cbwc[k][x][2:],
                              marker=marker[k], color=color[k], ms=10, mfc=color[k], elinewidth=1, lw=0,
                              ecolor=color[k], label=rm_label[k], alpha=0.5)
        ax[i, j].set_xlabel('Centrality class', fontsize=15)
        ax[i, j].set_xticks(x_range[2:])
        ax[i, j].set_xticklabels(x_arr[2:], minor=False, rotation=45)
        ax[i, j].set_ylabel(cr_labels[x], fontsize=15)
        ax[i, j].legend()
plt.show()
plt.close()

fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
x_range = np.linspace(1, len(cr_cbwc[0][0]), len(cr_cbwc[0][0]))
marker = ['o', '*', 'p', 's']
color = ['k', 'orange', 'blue', 'green']
for i in range(2):
    for j in range(2):
        x = i * 2 + j
        for k in (0, 3):
            ax[i, j].errorbar(x_range[2:]+(0.1*k), c_cbwc[k][x][2:], yerr=e_cbwc[k][x][2:],
                              marker=marker[k], color=color[k], ms=10, mfc=color[k], elinewidth=1, lw=0,
                              ecolor=color[k], label=rm_label[k], alpha=0.5)
        ax[i, j].set_xlabel('Centrality class', fontsize=15)
        ax[i, j].set_xticks(x_range[2:])
        ax[i, j].set_xticklabels(x_arr[2:], minor=False, rotation=45)
        ax[i, j].set_ylabel(labels[x], fontsize=15)
        ax[i, j].legend()
plt.show()
plt.close()
