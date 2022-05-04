# \author Skipper Kagamaster
# \date 01/10/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
#

"""
This code is made to display the correlations between
centrality metrics and found protons.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import reader as rdr

# Are you fitting to b, refmult1, or refmult3?
target = 'refmult3'
energy = 14
gev = '200'

if energy == 11:
    gev = '11'
if energy == 14:
    gev = '14.6'
if energy == 19:
    gev = '19.6'
if energy == 27:
    gev = '27.7'

# Let's load up the data.
data_directory = r"F:\UrQMD\14"
cent_directory = r'C:\Users\dansk\Documents\Thesis\Protons\2022_urqmd\pro_count_archives\ucc_events'
os.chdir(data_directory)
# os.chdir(cent_directory)  # To run on the previous version
pro_counts = np.load('pro_counts.npy', allow_pickle=True).astype('float')
pro_bins = np.load('pro_bins.npy', allow_pickle=True)

ylabels = [r'$X_{RM3}$', r'$X_{RM1}$', 'b (fm)', r'$X_{LW}$',
           r'$X_{ReLU}$', r'$X_{swish}$', r'$X_{CNN}$']
ylabels = [r'$X_{RM3}$', r'$X_{RM1}$', 'b (fm)', r'$X_{LW}$',
           r'$X_{ReLU}$', r'$X_{swish}$', r'$X_{CNN}$']
xlabels = ['protons', 'antiprotons', 'net protons']

# First we find the cumulants and their errors.
C = [[], [], [], []]
C_labels = [r'$C_1$', r'$C_2$', r'$C_3$', r'$C_4$']
E = [[], [], [], []]
E_rat = [[], [], [], []]
for k in range(4):
    for i in range(3):
        C[k].append([])
        E[k].append([])
        E_rat[k].append([])
        for j in range(len(ylabels)):
            if k == 0:  # The mean does not require the mean, so it's a different function.
                C[k][i].append(rdr.u_1(pro_counts[i][j], pro_bins[i][j]))
                C[k][i][j][C[k][i][j] == 0] = 1e-6
            elif k == 3:  # C4 is not simply the 4th moment, unlike C1-3.
                C[k][i].append(np.subtract(rdr.u_n(pro_counts[i][j], pro_bins[i][j], C[0][i][j], 4),
                                           3 * np.power(C[1][i][j], 2)))
                C[k][i][j][C[k][i][j] == 0] = 1e-6
            else:
                C[k][i].append(rdr.u_n(pro_counts[i][j], pro_bins[i][j], C[0][i][j], k + 1))
                C[k][i][j][C[k][i][j] == 0] = 1e-6
            E_rat[k][i].append(rdr.err_rat(pro_counts[i][j], pro_bins[i][j], C[0][i][j], power=(k + 1)))
            E[k][i].append(rdr.err(pro_counts[i][j], pro_bins[i][j], C[0][i][j], power=(k + 1)))
C = np.asarray(C, dtype='object')
E = np.asarray(E, dtype='object')
E_rat = np.asarray(E_rat, dtype='object')

# Now to plot the found cumulants with error.
# rdr.cum_plot_int_err(ylabels, target, pro_bins, C, E, C_labels, xlabels, gev=gev)

# 2D plot of the X values vs (anti/net)proton
xlim = ((0, 40), (0, 20), (-5, 40))
fig, ax = plt.subplots(3, len(ylabels), figsize=(16, 9), constrained_layout=True)
for i in range(3):
    for j in range(len(ylabels)):
        im = ax[i, j].pcolormesh(pro_bins[i][j][1], pro_bins[i][j][0], pro_counts[i][j], cmap='jet', norm=LogNorm())
        ax[i, j].set_ylabel(ylabels[j])
        ax[i, j].set_xlabel(xlabels[i])
        ax[i, j].set_xlim(xlim[i])
        fig.colorbar(im, ax=ax[i, j])
# plt.show()
plt.close()

# Now for ratios and CBWC, but without error for now.

C_up_test = C + E
C_down_test = C - E

Cc_up_test = np.copy(C_up_test)
Cc_down_test = np.copy(C_down_test)

arr_test = C[1] - E[1]
arr_test[arr_test == 0] = 1e-6
arr_test1 = C[0] - E[0]
arr_test1[arr_test1 == 0] = 1e-6

C_up_test[2] = np.divide(C[2] + E[2], arr_test)
C_up_test[3] = np.divide(C[3] + E[3], arr_test)
C_up_test[1] = np.divide(C[1] + E[1], arr_test1)
C_down_test[2] = np.divide(C[2] - E[2], arr_test)
C_down_test[3] = np.divide(C[3] - E[3], arr_test)
C_down_test[1] = np.divide(C[1] - E[1], arr_test1)

C_copy = np.copy(C)

C[2] = np.divide(C[2], C[1])
C[3] = np.divide(C[3], C[1])
C[1] = np.divide(C[1], C[0])

C_labels = [r'$\mu$', r'$\frac{\sigma^2}{\mu}$',
            r'$S\sigma$', r'$\kappa\sigma^2$']
# rdr.cum_plot_int_err(ylabels, target, pro_bins, C, E_rat, C_labels, xlabels, gev=gev)
# rdr.cum_plot_int_err_test(ylabels, target, pro_bins, C, C_up_test, C_down_test, C_labels, xlabels, gev=gev)

"""
Now to get the CBWC results.
"""
os.chdir(cent_directory)
pro_counts_c = np.load('pro_counts.npy', allow_pickle=True)
pro_bins_c = np.load('pro_bins.npy', allow_pickle=True)
centrality = []
cent_bins = np.hstack((np.linspace(10, 90, 9), (95, 97, 99)))
cent_bins_reverse = np.hstack(((1, 3, 5), np.linspace(10, 90, 9)))
centralities = ['75-100%', '50-75%', '25-50%', '0-25%']
centralities = ['95-100%', '90-95%', '85-90%', '80-85%', '75-80%', '70-75%',
                '65-70%', '60-65%', '55-60%', '50-55%', '45-50%', '40-45%',
                '35-40%', '30-35%', '25-30%', '20-25%', '15-20%', '10-15%',
                '5-10%', '2-5%', '1-2%', '0-1%']
centralities = ['90-100%', '80-90%', '70-80%', '60-70%', '50-60%', '40-50%',
                '30-40%', '20-30%', '10-20%', '5-10%', '3-5%', '1-3%', '0-1%']

cent_bins = np.hstack((np.linspace(10, 90, 9), 95))
cent_bins_reverse = np.hstack((5, np.linspace(10, 90, 9)))
centralities = ["90-100%", "80-90%", "70-80%", "60-70%", "50-60%", "40-50%", "30-40%",
                "20-30%", "10-20%", "5-10%", "0-5%"]

for i in range(len(ylabels)):
    if target == 'b':
        if i > 1:
            centrality.append(rdr.centrality(pro_counts_c[0][i], pro_bins_c[0][i],
                                             reverse=True, cent_bins_reverse=cent_bins_reverse))
        else:
            centrality.append(rdr.centrality(pro_counts_c[0][i], pro_bins_c[0][i], cent_bins=cent_bins))
    else:
        if i == 2:
            centrality.append(rdr.centrality(pro_counts_c[0][i], pro_bins_c[0][i],
                                             reverse=True, cent_bins_reverse=cent_bins_reverse))
        else:
            centrality.append(rdr.centrality(pro_counts_c[0][i], pro_bins_c[0][i], cent_bins=cent_bins))
print(centrality)

# Centrality taking away the RefMult3 == 0 events as they have some weird pileup.
centrality = [[6., 14., 28., 50., 84., 131., 197., 285., 408., 492.],
              [3., 7., 15., 27., 45., 72., 108., 157., 227., 274.],
              [3.19, 4.51, 6.38, 7.8, 9.01, 10.07, 11.03, 11.92, 12.75, 13.61],
              [4.338, 13.958, 29.858, 55.32, 92.322, 142.58, 207.161, 289.727, 394.215, 457.208],
              [6.503, 13.531, 27.266, 49.652, 83.318, 131.64, 196.276, 283.764, 404.833, 485.149],
              [6.076, 13.573, 27.286, 49.105, 83.39, 131.38, 195.412, 283.929, 403.789, 484.684],
              [4.991, 12.16, 25.672, 46.537, 79.563, 126.518, 192.391, 281.273, 401.452, 481.841]]

os.chdir(data_directory)
C_cbwc = []
E_cbwc = []
E_u_cbwc = []
E_d_cbwc = []

# This is to not use CBWC for comparison
C_no_cbwc = []
E_no_cbwc = []
E_u_no_cbwc = []
E_d_no_cbwc = []

# These are for the non-ratio cumulants (i.e. no volume correction).
Cc_cbwc = []
Ec_cbwc = []
Ec_u_cbwc = []
Ec_d_cbwc = []

# And for non-ratio, non-cbwc
Cc_no_cbwc = []
Ec_no_cbwc = []
Ec_u_no_cbwc = []
Ec_d_no_cbwc = []

for k in range(4):
    C_cbwc.append([])
    E_cbwc.append([])
    E_u_cbwc.append([])
    E_d_cbwc.append([])

    C_no_cbwc.append([])
    E_no_cbwc.append([])
    E_u_no_cbwc.append([])
    E_d_no_cbwc.append([])

    Cc_cbwc.append([])
    Ec_cbwc.append([])
    Ec_u_cbwc.append([])
    Ec_d_cbwc.append([])

    Cc_no_cbwc.append([])
    Ec_no_cbwc.append([])
    Ec_u_no_cbwc.append([])
    Ec_d_no_cbwc.append([])

    for i in range(3):
        C_cbwc[k].append([])
        E_cbwc[k].append([])
        E_u_cbwc[k].append([])
        E_d_cbwc[k].append([])

        C_no_cbwc[k].append([])
        E_no_cbwc[k].append([])
        E_u_no_cbwc[k].append([])
        E_d_no_cbwc[k].append([])

        Cc_cbwc[k].append([])
        Ec_cbwc[k].append([])
        Ec_u_cbwc[k].append([])
        Ec_d_cbwc[k].append([])

        Cc_no_cbwc[k].append([])
        Ec_no_cbwc[k].append([])
        Ec_u_no_cbwc[k].append([])
        Ec_d_no_cbwc[k].append([])

        for j in range(len(ylabels)):
            C_cbwc[k][i].append(rdr.cbwc(pro_counts[i][j],
                                         pro_bins[i][j],
                                         C[k][i][j],
                                         centrality[j]))
            E_cbwc[k][i].append(rdr.cbwc(pro_counts[i][j],
                                         pro_bins[i][j],
                                         E_rat[k][i][j],
                                         centrality[j]))
            E_u_cbwc[k][i].append(rdr.cbwc(pro_counts[i][j],
                                           pro_bins[i][j],
                                           C_up_test[k][i][j],
                                           centrality[j]))
            E_d_cbwc[k][i].append(rdr.cbwc(pro_counts[i][j],
                                           pro_bins[i][j],
                                           C_down_test[k][i][j],
                                           centrality[j]))

            C_no_cbwc[k][i].append(rdr.no_cbwc(pro_counts[i][j],
                                               pro_bins[i][j],
                                               C[k][i][j],
                                               centrality[j]))
            E_no_cbwc[k][i].append(rdr.no_cbwc(pro_counts[i][j],
                                               pro_bins[i][j],
                                               E_rat[k][i][j],
                                               centrality[j]))
            E_u_no_cbwc[k][i].append(rdr.no_cbwc(pro_counts[i][j],
                                                 pro_bins[i][j],
                                                 C_up_test[k][i][j],
                                                 centrality[j]))
            E_d_no_cbwc[k][i].append(rdr.no_cbwc(pro_counts[i][j],
                                                 pro_bins[i][j],
                                                 C_down_test[k][i][j],
                                                 centrality[j]))

            Cc_cbwc[k][i].append(rdr.cbwc(pro_counts[i][j],
                                          pro_bins[i][j],
                                          C_copy[k][i][j],
                                          centrality[j]))
            Ec_cbwc[k][i].append(rdr.cbwc(pro_counts[i][j],
                                          pro_bins[i][j],
                                          E[k][i][j],
                                          centrality[j]))
            Ec_u_cbwc[k][i].append(rdr.cbwc(pro_counts[i][j],
                                            pro_bins[i][j],
                                            Cc_up_test[k][i][j],
                                            centrality[j]))
            Ec_d_cbwc[k][i].append(rdr.cbwc(pro_counts[i][j],
                                            pro_bins[i][j],
                                            Cc_down_test[k][i][j],
                                            centrality[j]))
            Cc_no_cbwc[k][i].append(rdr.no_cbwc(pro_counts[i][j],
                                                pro_bins[i][j],
                                                C_copy[k][i][j],
                                                centrality[j]))
            Ec_no_cbwc[k][i].append(rdr.no_cbwc(pro_counts[i][j],
                                                pro_bins[i][j],
                                                E[k][i][j],
                                                centrality[j]))
            Ec_u_no_cbwc[k][i].append(rdr.no_cbwc(pro_counts[i][j],
                                                  pro_bins[i][j],
                                                  Cc_up_test[k][i][j],
                                                  centrality[j]))
            Ec_d_no_cbwc[k][i].append(rdr.no_cbwc(pro_counts[i][j],
                                                  pro_bins[i][j],
                                                  Cc_down_test[k][i][j],
                                                  centrality[j]))

C_E_up_cbwc = np.add(np.asarray(C_cbwc), np.asarray(E_cbwc))
C_E_down_cbwc = np.subtract(np.asarray(C_cbwc), np.asarray(E_cbwc))

Cc_E_up_cbwc = np.add(np.asarray(Cc_cbwc), np.asarray(Ec_cbwc))
Cc_E_down_cbwc = np.subtract(np.asarray(Cc_cbwc), np.asarray(Ec_cbwc))

# Now to plot all the CBWC values
markers = ['o', '8', r'$b$', 's', 'P', '*', 'X', 'D']
color = ['orangered', 'orange', 'sienna', 'black', 'blue', 'purple', 'darkviolet', 'royalblue']
for k in range(4):
    fig, ax = plt.subplots(3, len(ylabels), figsize=(16, 9), constrained_layout=True)
    plt.suptitle(C_labels[k] + r' for $\sqrt{s_{NN}}$= ' + gev + ' GeV (UrQMD)',
                 fontsize=30)
    for i in range(3):
        for j in range(len(ylabels)):
            if target == 'b':
                if j > 1:
                    ax[i, j].fill_between(centralities, C_E_up_cbwc[k][i][j][::-1],
                                          C_E_down_cbwc[k][i][j][::-1], facecolor=color[j],
                                          alpha=0.2)
                    ax[i, j].scatter(centralities, C_cbwc[k][i][j][::-1], marker=markers[j], s=50, c=color[j])
                    ax[i, j].set_xticks(ax[i, j].get_xticks())
                    ax[i, j].set_xticklabels(centralities, rotation=90)
                else:
                    ax[i, j].fill_between(centralities, C_E_up_cbwc[k][i][j],
                                          C_E_down_cbwc[k][i][j], facecolor=color[j],
                                          alpha=0.2)
                    ax[i, j].scatter(centralities, C_cbwc[k][i][j], marker=markers[j], s=50, c=color[j])
                    ax[i, j].set_xticks(ax[i, j].get_xticks())
                    ax[i, j].set_xticklabels(centralities, rotation=90)
                ax[i, j].set_xlabel(ylabels[j])
                ax[i, j].set_ylabel(r'<' + xlabels[i] + r'>')
            else:
                if j == 2:
                    ax[i, j].fill_between(centralities, Cc_E_up_cbwc[k][i][j][::-1],
                                          Cc_E_down_cbwc[k][i][j][::-1], facecolor=color[j],
                                          alpha=0.2)
                    ax[i, j].scatter(centralities, Cc_cbwc[k][i][j][::-1], marker=markers[j], s=50, c=color[j])
                    ax[i, j].set_xticks(ax[i, j].get_xticks())
                    ax[i, j].set_xticklabels(centralities, rotation=90)
                else:
                    ax[i, j].fill_between(centralities, Cc_E_up_cbwc[k][i][j],
                                          Cc_E_down_cbwc[k][i][j], facecolor=color[j],
                                          alpha=0.2)
                    ax[i, j].scatter(centralities, Cc_cbwc[k][i][j], marker=markers[j], s=50, c=color[j])
                    ax[i, j].set_xticks(ax[i, j].get_xticks())
                    ax[i, j].set_xticklabels(centralities, rotation=90)
                ax[i, j].set_xlabel(ylabels[j])
                ax[i, j].set_ylabel(r'<' + xlabels[i] + r'>')
    plt.show()
    plt.close()

# This is the end, final plot. Are there ACEs shown?
plt_set = (0, 1, 2, 3, 6)
plt.figure(figsize=(16, 9))
plt.title(r'$\kappa\sigma^2$ for $\sqrt{s_{NN}}$ = ' + gev + ' GeV (UrQMD)', fontsize=30)
for j in plt_set:
    if target == 'b':
        if j > 1:
            plt.scatter(centralities, C_cbwc[-1][-1][j][::-1], marker=markers[j], s=50, alpha=0.5, c=color[j],
                        label=ylabels[j])
        else:
            plt.scatter(centralities, C_cbwc[-1][-1][j], marker=markers[j], s=50, alpha=0.5, c=color[j],
                        label=ylabels[j])
    else:
        if j == 2:
            plt.scatter(centralities, C_cbwc[-1][-1][j][::-1], marker=markers[j], s=50, alpha=0.5, c=color[j],
                        label=ylabels[j])
        else:
            plt.scatter(centralities, C_cbwc[-1][-1][j], marker=markers[j], s=50, alpha=0.5, c=color[j],
                        label=ylabels[j])
plt.xticks(centralities, labels=centralities, rotation=45)
plt.xlabel("Centrality", fontsize=20)
plt.ylabel(r'$\kappa\sigma^2$', fontsize=20)
plt.legend()
# plt.show()
plt.close()

# And with error (such that it is, and only some values).
plt.figure(figsize=(9, 5))
# plt.title(r'$\frac{\sigma^2}{\mu}$ for $\sqrt{s_{NN}}$ = ' + gev + ' GeV (UrQMD)', fontsize=30)

fill_up = np.add(C_cbwc, E_cbwc)
fill_down = np.subtract(C_cbwc, E_cbwc)

for j in plt_set:
    if target == 'b':
        if j > 1:
            plt.errorbar(centralities, C_cbwc[1][-1][j][::-1],
                         E_cbwc[1][-1][j][::-1],
                         marker=markers[j], ms=10, alpha=0.5, c=color[j],
                         label=ylabels[j], lw=0, elinewidth=1, capsize=2)
        else:
            plt.errorbar(centralities, C_cbwc[1][-1][j],
                         E_cbwc[1][-1][j],
                         marker=markers[j], ms=10, alpha=0.5, c=color[j],
                         label=ylabels[j], lw=0, elinewidth=1, capsize=2)
    else:
        if j == 2:
            plt.errorbar(centralities, C_cbwc[1][-1][j][::-1],
                         E_cbwc[1][-1][j][::-1],
                         marker=markers[j], ms=10, alpha=0.5, c=color[j],
                         label=ylabels[j], lw=0, elinewidth=1, capsize=2)
        else:
            plt.errorbar(centralities, C_cbwc[1][-1][j],
                         E_cbwc[1][-1][j],
                         marker=markers[j], ms=10, alpha=0.5, c=color[j],
                         label=ylabels[j], lw=0, elinewidth=1, capsize=2)
plt.xticks(centralities, labels=centralities, rotation=45)
plt.xlabel("Centrality", fontsize=20)
plt.ylabel(r'$\frac{\sigma^2}{\mu}$', fontsize=20)
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

# This is the one I'm modifying!!!
plt.figure(figsize=(12, 7), constrained_layout=True)
# plt.title(r'$\frac{\sigma^2}{\mu}$ for $\sqrt{s_{NN}}$ = ' + gev + ' GeV (UrQMD)', fontsize=30)
size = 50
alpha = 0.5
for j in plt_set:
    if target == 'b':
        if j > 1:
            plt.fill_between(centralities, fill_up[1][-1][j][::-1], fill_down[1][-1][j][::-1], alpha=0.2,
                             color=color[j])
            plt.scatter(centralities, C_cbwc[1][-1][j][::-1],
                        marker=markers[j], s=size, alpha=alpha, c=color[j],
                        label=ylabels[j])
        else:
            plt.fill_between(centralities, fill_up[1][-1][j], fill_down[1][-1][j], alpha=0.2,
                             color=color[j])
            plt.scatter(centralities, C_cbwc[1][-1][j],
                        marker=markers[j], s=size, alpha=alpha, c=color[j],
                        label=ylabels[j])
    else:
        if j == 2:
            plt.fill_between(centralities, fill_up[1][-1][j][::-1], fill_down[1][-1][j][::-1], alpha=0.2,
                             color=color[j])
            plt.scatter(centralities, C_cbwc[1][-1][j][::-1],
                        marker=markers[j], s=size, alpha=alpha, c=color[j],
                        label=ylabels[j])
        else:
            plt.fill_between(centralities, fill_up[1][-1][j], fill_down[1][-1][j], alpha=0.2,
                             color=color[j])
            plt.scatter(centralities, C_cbwc[1][-1][j],
                        marker=markers[j], s=size, alpha=alpha, c=color[j],
                        label=ylabels[j])
plt.xticks(centralities, labels=centralities, rotation=45)
plt.xlabel("Centrality", fontsize=20)
plt.ylabel(r'$\frac{\sigma^2}{\mu}$', fontsize=20)
plt.legend(fontsize=15)
# plt.tight_layout()
plt.show()

# Doing some nonsense here to show why we care about cbwc and errorbars.
ylabels_nocbwc = [r'$X_{RM3}$ no cbwc', r'$X_{RM1}$ no cbwc', 'b (fm) no cbwc',
                  r'$X_{LW}$ no cbwc', r'$X_{ReLU}$ no cbwc', r'$X_{swish}$ no cbwc',
                  r'$X_{CNN}$ no cbwc']
markers = ['o', '8', r'$b$', 's', 'P', '*', 'X', 'D']
color_nocbwc = ['purple', 'orange', 'black', 'black', 'orange',
                'purple', 'darkviolet', 'royalblue']
mar_s = 200
plt_set = (0, 2, 4)
plt.figure(figsize=(16, 9))
# plt.title(r'$\mu$ for $\sqrt{s_{NN}}$=19.6 GeV, UrQMD', fontsize=30)
metric = -3
names = [r'$\mu$', r'$\sigma^2$', r'$S$', r'$\kappa']
for j in plt_set:
    if target == 'b':
        if j > 1:
            plt.scatter(centralities, Cc_cbwc[metric][-1][j][::-1],
                        marker=markers[j], s=mar_s, alpha=0.5, c=color[j],
                        label=ylabels[j], lw=0)
            plt.scatter(np.linspace(0.2, 10.2, 11), Cc_no_cbwc[metric][-1][j][::-1],
                        marker=markers[j], s=mar_s, alpha=0.5, c=color_nocbwc[j],
                        label=ylabels_nocbwc[j], lw=0)
        else:
            plt.scatter(centralities, Cc_cbwc[metric][-1][j],
                        marker=markers[j], s=mar_s, alpha=0.5, c=color[j],
                        label=ylabels[j], lw=0)
            plt.scatter(np.linspace(0.2, 10.2, 11), Cc_no_cbwc[metric][-1][j],
                        marker=markers[j], s=mar_s, alpha=0.5, c=color_nocbwc[j],
                        label=ylabels_nocbwc[j], lw=0)
    else:
        if j == 2:
            plt.scatter(centralities, Cc_cbwc[metric][-1][j][::-1],
                        marker=markers[j], s=mar_s, alpha=0.5, c=color[j],
                        label=ylabels[j], lw=0)
            plt.scatter(np.linspace(0.2, 10.2, 11), Cc_no_cbwc[metric][-1][j][::-1],
                        marker=markers[j], s=mar_s, alpha=0.5, c=color_nocbwc[j],
                        label=ylabels_nocbwc[j], lw=0)
        else:
            plt.scatter(centralities, Cc_cbwc[metric][-1][j],
                        marker=markers[j], s=mar_s, alpha=0.5, c=color[j],
                        label=ylabels[j], lw=0)
            plt.scatter(np.linspace(0.2, 10.2, 11), Cc_no_cbwc[metric][-1][j],
                        marker=markers[j], s=mar_s, alpha=0.5, c=color_nocbwc[j],
                        label=ylabels_nocbwc[j], lw=0)
plt.xticks(centralities, labels=centralities, rotation=45)
plt.xlabel("Centrality", fontsize=30)
plt.ylabel(names[metric], fontsize=30)
plt.legend(fontsize=30)
plt.show()
plt.figure(figsize=(16, 9))
# plt.title(r'$\mu$ for $\sqrt{s_{NN}}$=19.6 GeV, UrQMD', fontsize=30)
for j in plt_set:
    if target == 'b':
        if j > 1:
            plt.errorbar(centralities, Cc_cbwc[metric][-1][j][::-1],
                         Ec_cbwc[metric][-1][j][::-1],
                         marker=markers[j], ms=20, alpha=0.5, c=color[j],
                         label=ylabels[j], lw=0, elinewidth=1, capsize=2)
            plt.errorbar(np.linspace(0.2, 10.2, 11), Cc_no_cbwc[metric][-1][j][::-1],
                         Ec_no_cbwc[metric][-1][j][::-1],
                         marker=markers[j], ms=20, alpha=0.5, c=color_nocbwc[j],
                         label=ylabels_nocbwc[j], lw=0, elinewidth=1, capsize=2)
        else:
            plt.errorbar(centralities, Cc_cbwc[metric][-1][j],
                         Ec_cbwc[metric][-1][j],
                         marker=markers[j], ms=20, alpha=0.5, c=color[j],
                         label=ylabels[j], lw=0, elinewidth=1, capsize=2)
            plt.errorbar(np.linspace(0.2, 10.2, 11), Cc_no_cbwc[metric][-1][j],
                         Ec_no_cbwc[metric][-1][j],
                         marker=markers[j], ms=20, alpha=0.5, c=color_nocbwc[j],
                         label=ylabels_nocbwc[j], lw=0, elinewidth=1, capsize=2)
    else:
        if j == 2:
            plt.errorbar(centralities, Cc_cbwc[metric][-1][j][::-1],
                         Ec_cbwc[metric][-1][j][::-1],
                         marker=markers[j], ms=20, alpha=0.5, c=color[j],
                         label=ylabels[j], lw=0, elinewidth=1, capsize=2)
            plt.errorbar(np.linspace(0.2, 10.2, 11), Cc_no_cbwc[metric][-1][j][::-1],
                         Ec_no_cbwc[metric][-1][j][::-1],
                         marker=markers[j], ms=20, alpha=0.5, c=color_nocbwc[j],
                         label=ylabels_nocbwc[j], lw=0, elinewidth=1, capsize=2)
        else:
            plt.errorbar(centralities, Cc_cbwc[metric][-1][j],
                         Ec_cbwc[metric][-1][j],
                         marker=markers[j], ms=20, alpha=0.5, c=color[j],
                         label=ylabels[j], lw=0, elinewidth=1, capsize=2)
            plt.errorbar(np.linspace(0.2, 10.2, 11), Cc_no_cbwc[metric][-1][j],
                         Ec_no_cbwc[metric][-1][j],
                         marker=markers[j], ms=20, alpha=0.5, c=color_nocbwc[j],
                         label=ylabels_nocbwc[j], lw=0, elinewidth=1, capsize=2)
plt.xticks(centralities, labels=centralities, rotation=45)
plt.xlabel("Centrality", fontsize=30)
plt.ylabel(names[metric], fontsize=30)
plt.legend(fontsize=30)
plt.show()

# And with error (such that it is, and only some values).
plt.figure(figsize=(16, 9))
plt.title(r'$S\sigma$ for $\sqrt{s_{NN}}$ = ' + gev + ' GeV (UrQMD)', fontsize=30)
for j in plt_set:
    if target == 'b':
        if j > 1:
            plt.errorbar(centralities, C_cbwc[2][-1][j][::-1],
                         E_cbwc[2][-1][j][::-1],
                         marker=markers[j], ms=10, alpha=0.5, c=color[j],
                         label=ylabels[j], lw=0, elinewidth=1, capsize=2)
        else:
            plt.errorbar(centralities, C_cbwc[2][-1][j],
                         E_cbwc[2][-1][j],
                         marker=markers[j], ms=10, alpha=0.5, c=color[j],
                         label=ylabels[j], lw=0, elinewidth=1, capsize=2)
    else:
        if j == 2:
            plt.errorbar(centralities, C_cbwc[2][-1][j][::-1],
                         E_cbwc[2][-1][j][::-1],
                         marker=markers[j], ms=10, alpha=0.5, c=color[j],
                         label=ylabels[j], lw=0, elinewidth=1, capsize=2)
        else:
            plt.errorbar(centralities, C_cbwc[2][-1][j],
                         E_cbwc[2][-1][j],
                         marker=markers[j], ms=10, alpha=0.5, c=color[j],
                         label=ylabels[j], lw=0, elinewidth=1, capsize=2)
plt.xticks(centralities, labels=centralities, rotation=45)
plt.xlabel("Centrality", fontsize=20)
plt.ylabel(r'$S\sigma$', fontsize=20)
plt.legend()
plt.show()

# And with error (such that it is, and only some values).
plt.figure(figsize=(16, 9))
plt.title(r'$\kappa\sigma^2$ for $\sqrt{s_{NN}}$ = ' + gev + ' GeV (UrQMD)', fontsize=30)
for j in plt_set:
    if target == 'b':
        if j > 1:
            plt.errorbar(centralities, C_cbwc[-1][-1][j][::-1],
                         E_cbwc[-1][-1][j][::-1],
                         marker=markers[j], ms=10, alpha=0.5, c=color[j],
                         label=ylabels[j], lw=0, elinewidth=1, capsize=2)
        else:
            plt.errorbar(centralities, C_cbwc[-1][-1][j],
                         E_cbwc[-1][-1][j],
                         marker=markers[j], ms=10, alpha=0.5, c=color[j],
                         label=ylabels[j], lw=0, elinewidth=1, capsize=2)
    else:
        if j == 2:
            plt.errorbar(centralities, C_cbwc[-1][-1][j][::-1],
                         E_cbwc[-1][-1][j][::-1],
                         marker=markers[j], ms=10, alpha=0.5, c=color[j],
                         label=ylabels[j], lw=0, elinewidth=1, capsize=2)
        else:
            plt.errorbar(centralities, C_cbwc[-1][-1][j],
                         E_cbwc[-1][-1][j],
                         marker=markers[j], ms=10, alpha=0.5, c=color[j],
                         label=ylabels[j], lw=0, elinewidth=1, capsize=2)
plt.xticks(centralities, labels=centralities, rotation=45)
plt.xlabel("Centrality", fontsize=20)
plt.ylabel(r'$\kappa\sigma^2$', fontsize=20)
plt.legend()
plt.show()
