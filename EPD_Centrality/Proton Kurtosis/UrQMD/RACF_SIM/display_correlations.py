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
import copy

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
xlabels = ['protons', 'antiprotons', 'net protons']
EPD_set = [3, 4, 5, 6]
Final_set = [0, 2, 3, 4]

# First we find the cumulants and their errors.
C = [[], [], [], []]
C_labels = [r'$C_1$', r'$C_2$', r'$C_3$', r'$C_4$']
C_labels_rat = [r'$C_1$', r'$\frac{C_2}{C_1}$', r'$\frac{C_3}{C_2}$', r'$\frac{C_4}{C_2}$']
E = [[], [], [], []]
E_rat = [[], [], [], []]
E_rat_min = [[], [], [], []]  # This is for min/max errors on the ratios.
E_rat_max = [[], [], [], []]  # This is for min/max errors on the ratios.
p_bins = [[], [], [], []]
n_bins = [[], [], [], []]
for k in range(4):
    for i in range(3):
        C[k].append([])
        E[k].append([])
        p_bins[k].append([])
        E_rat[k].append([])
        E_rat_min[k].append([])
        E_rat_max[k].append([])
        n_bins[k].append([])
        for j in range(len(ylabels)):
            # Moments for cumulants and errors.
            u = []
            x = []
            n = []
            for r in range(1, 9):
                u_temp = rdr.u_n(pro_counts[i][j], pro_bins[i][j], r)
                u.append(u_temp[0])
                x.append(u_temp[1])
                n.append(u_temp[2])
                index = np.where(u[r - 1] != 0)
                if r == 2:
                    index = np.where(u[r - 1] != 0)
                    u[r - 1] = u[r - 1][index]
                    x[r - 1] = x[r - 1][index]
                    n[r - 1] = n[r - 1][index]
            # This is to make sure we don't have different sized arrays
            # (i.e. that all x values are the same set for a given X,
            # and that no n=0 are allowed).
            x_full = x[0]
            for r in range(len(x)):
                index = np.where(n[r] != 0)
                n[r] = n[r][index]
                x[r] = x[r][index]
                u[r] = u[r][index]
                x_full = np.intersect1d(x_full, x[r])
            for r in range(len(x)):
                index = []
                for g in range(len(x[r])):
                    if x[r][g] in x_full:
                        index.append(g)
                x[r] = x[r][index]
                u[r] = u[r][index]
                n[r] = n[r][index]
            u2 = (-u[1] ** 2 + u[3])
            u3 = 9 * (u[1] ** 3) - 6 * u[1] * u[3] - u[2] ** 2 + u[5]
            u4 = -36 * (u[1] ** 4) + 48 * (u[1] ** 2) * u[3] + 64 * (u[4] ** 2) * u[1] - 12 * u[1] * u[5] \
                 - 8 * u[2] * u[4] - u[3] ** 2 + u[7]
            n_mean = u[0]
            ur2 = np.divide(np.subtract(u[3], np.power(u[1], 2)), np.power(n_mean, 2)) - \
                  np.divide(2 * np.multiply(u[1], u[2]), np.power(n_mean, 3)) + \
                  np.divide(np.power(u[1], 3), np.power(n_mean, 4))
            ur3 = 9 * np.power(u[1], 2) - \
                  np.divide(6 * u[3], u[1]) + \
                  np.divide(np.add(6 * u[2], u[5]), np.power(u[1], 2)) - \
                  np.divide(np.multiply(2 * u[2], u[4]), np.power(u[1], 3)) + \
                  np.divide(np.multiply(np.power(u[2], 2), u[3]), np.power(u[1], 4))
            ur4 = -9 * np.power(u[1], 2) + 9 * u[3] + \
                  np.divide(np.subtract(40 * np.power(u[2], 2), 6 * u[5]), u[1]) + \
                  np.divide(np.subtract(np.add(u[7], 6 * np.power(u[3], 2)),
                                        8 * np.multiply(u[2], u[4])),
                            np.power(u[1], 2)) + \
                  np.divide(np.subtract(8 * np.multiply(np.power(u[2], 4), u[3]),
                                        2 * np.multiply(u[3], u[5])),
                            np.power(u[1], 3)) + \
                  np.divide(np.power(u[3], 3), np.power(u[1], 4))
            index = np.where((u[1] > 0) & (u2 > 0) & (u3 > 0) & (u4 > 0) &
                             (ur2 > 0) & (ur3 > 0) & (ur4 > 0))
            u2 = u2[index]
            u3 = u3[index]
            u4 = u4[index]
            ur2 = ur2[index]
            ur3 = ur3[index]
            ur4 = ur4[index]
            for r in range(len(x)):
                x[r] = x[r][index]
                u[r] = u[r][index]
                n[r] = n[r][index]
            u_er = [u[1], u2, u3, u4]
            u_er_rat = [u[1], ur2, ur3, ur4]
            # Now to generate the cumulants, their ratios, and errors.
            if k == 3:  # C4 is not simply the 4th moment, unlike C1-3.
                p_bins[k][i].append(x[k])
                n_bins[k][i].append(n[k])
                C[k][i].append(np.subtract(u[3], 3 * np.power(u[1], 2)))
                E[k][i].append(np.sqrt(np.divide(u_er[k], n[k])))
                E_rat[k][i].append(np.sqrt(np.divide(u_er_rat[k], n[k])))
            else:
                C[k][i].append(u[k])
                E[k][i].append(np.sqrt(np.divide(u_er[k], n[k])))
                E_rat[k][i].append(np.sqrt(np.divide(u_er_rat[k], n[k])))
                p_bins[k][i].append(x[k])
                n_bins[k][i].append(n[k])
                index = C[k][i][j] != 0
                C[k][i][j] = C[k][i][j][index]
                p_bins[k][i][j] = p_bins[k][i][j][index]
                n_bins[k][i][j] = n_bins[k][i][j][index]
                E[k][i][j] = E[k][i][j][index]
                E_rat[k][i][j] = E_rat[k][i][j][index]

# This bit of code makes sure all cumulants have the same x coordinates (e.g. not
# a seperate length for the antiproton and net proton cumulant arrays, or a seperate
# length for C4 vs C2).
len_check = copy.deepcopy(p_bins)
for k in range(4):
    for j in range(len(ylabels)):
        new_p = p_bins[k][0][j]
        for i in range(3):
            new_p = np.intersect1d(new_p, p_bins[k][i][j])
        for i in range(3):
            index = []
            for r in range(len(p_bins[k][i][j])):
                if p_bins[k][i][j][r] in new_p:
                    index.append(r)
            C[k][i][j] = C[k][i][j][index]
            p_bins[k][i][j] = p_bins[k][i][j][index]
            n_bins[k][i][j] = n_bins[k][i][j][index]
            E[k][i][j] = E[k][i][j][index]
            E_rat[k][i][j] = E_rat[k][i][j][index]
len_check = copy.deepcopy(p_bins)
for i in range(3):
    for j in range(len(ylabels)):
        new_p = p_bins[0][i][j]
        for k in range(4):
            new_p = np.intersect1d(new_p, p_bins[k][i][j])
        for k in range(4):
            index = []
            for r in range(len(p_bins[k][i][j])):
                if p_bins[k][i][j][r] in new_p:
                    index.append(r)
            C[k][i][j] = C[k][i][j][index]
            p_bins[k][i][j] = p_bins[k][i][j][index]
            n_bins[k][i][j] = n_bins[k][i][j][index]
            E[k][i][j] = E[k][i][j][index]
            E_rat[k][i][j] = E_rat[k][i][j][index]

# Now to make an array for errors on the ratio using the min/max approach.
for j in range(3):
    for k in range(len(ylabels)):
        E_rat_min[0][j].append(np.subtract(C[0][j][k], E[0][j][k]))
        E_rat_max[0][j].append(np.add(C[0][j][k], E[0][j][k]))
        E_rat_max[1][j].append(np.divide(np.add(C[1][j][k], E[1][j][k]), np.subtract(C[0][j][k], E[0][j][k])))
        E_rat_min[1][j].append(np.divide(np.subtract(C[1][j][k], E[1][j][k]), np.add(C[0][j][k], E[0][j][k])))
        E_rat_max[2][j].append(np.divide(np.add(C[2][j][k], E[2][j][k]), np.subtract(C[1][j][k], E[1][j][k])))
        E_rat_min[2][j].append(np.divide(np.subtract(C[2][j][k], E[2][j][k]), np.add(C[1][j][k], E[1][j][k])))
        E_rat_max[3][j].append(np.divide(np.add(C[3][j][k], E[3][j][k]), np.subtract(C[1][j][k], E[1][j][k])))
        E_rat_min[3][j].append(np.divide(np.subtract(C[3][j][k], E[3][j][k]), np.add(C[1][j][k], E[1][j][k])))
C = np.asarray(C, dtype='object')
E = np.asarray(E, dtype='object')
E_rat = np.asarray(E_rat, dtype='object')
p_bins = np.asarray(p_bins, dtype='object')
n_bins = np.asarray(n_bins, dtype='object')
C_rat = np.copy(C)
for j in range(3):
    for k in range(len(ylabels)):
        C_rat[2][j][k] = np.divide(C_rat[2][j][k], C_rat[1][j][k])
        C_rat[3][j][k] = np.divide(C_rat[3][j][k], C_rat[1][j][k])
        C_rat[1][j][k] = np.divide(C_rat[1][j][k], C_rat[0][j][k])

# Let's try a plot with the errors plotted alongside each other.
color = ['orangered', 'orange', 'sienna', 'black', 'blue', 'purple', 'darkviolet', 'royalblue']
markers = ['o', '8', r'$b$', 's', 'P', '*', 'X', 'D']
for i in range(4):
    fig, ax = plt.subplots(3, len(ylabels), figsize=(16, 9), constrained_layout=True)
    for j in range(3):
        for k in range(len(ylabels)):
            r = j * len(ylabels)
            if k == 2:
                ax[j][k].fill_between(p_bins[i][j][k], E_rat_max[i][j][k][::-1], E_rat_min[i][j][k][::-1],
                                      color=color[k])
                # ax[j][k].fill_between(p_bins[i][j][k], C_rat[i][j][k][::-1]+E_rat[i][j][k][::-1],
                # C_rat[i][j][k][::-1]-E_rat[i][j][k][::-1],
                #                      color='k', alpha=0.7)
                ax[j][k].plot(p_bins[i][j][k], C_rat[i][j][k][::-1], color='k', marker=markers[k], ms=1, lw=0)
            else:
                ax[j][k].fill_between(p_bins[i][j][k], E_rat_max[i][j][k], E_rat_min[i][j][k],
                                      color=color[k])
                # ax[j][k].fill_between(p_bins[i][j][k], C_rat[i][j][k]+E_rat[i][j][k],
                # C_rat[i][j][k]-E_rat[i][j][k],
                #                      color='k', alpha=0.7)
                ax[j][k].plot(p_bins[i][j][k], C_rat[i][j][k], color='k', marker=markers[k], ms=1, lw=0)
            ax[j][k].set_xlabel(ylabels[k], fontsize=15)
    fig.suptitle(C_labels_rat[i], fontsize=20)
    # plt.show()
    plt.close()

# Now to plot the found cumulants with error.
# rdr.cum_plot_int_err(ylabels, target, p_bins, C, E, C_labels, xlabels, gev=gev)
# rdr.cum_plot_int_err(ylabels, target, p_bins, C_rat, E_rat, C_labels_rat, xlabels, gev=gev)

# 2D plot of the X values vs (anti/net)proton
'''
xlim = ((0, 40), (0, 20), (-5, 40))
fig, ax = plt.subplots(3, len(ylabels), figsize=(16, 9), constrained_layout=True)
for i in range(3):
    for j in range(len(ylabels)):
        im = ax[i, j].pcolormesh(pro_bins[i][j][1], pro_bins[i][j][0], pro_counts[i][j], cmap='jet', norm=LogNorm())
        ax[i, j].set_ylabel(ylabels[j])
        ax[i, j].set_xlabel(xlabels[i])
        ax[i, j].set_xlim(xlim[i])
        fig.colorbar(im, ax=ax[i, j])
plt.show()
plt.close()
'''
# Now for ratios and CBWC, but without error for now.
'''Turned off for now.'''
'''
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
'''
C_labels = [r'$\mu$', r'$\frac{\sigma^2}{\mu}$',
            r'$S\sigma$', r'$\kappa\sigma^2$']
# rdr.cum_plot_int_err(ylabels, target, p_bins, C, E_rat, C_labels, xlabels, gev=gev)
# rdr.cum_plot_int_err_test(ylabels, target, p_bins, C, C_up_test, C_down_test, C_labels, xlabels, gev=gev)

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
                '30-40%', '20-30%', '10-20%', '5-10%', '0-5%']

cent_bins = np.hstack((np.linspace(10, 90, 9), 95))
cent_bins_reverse = np.hstack((5, np.linspace(10, 90, 9)))

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

# Centrality taking away the RefMult3 == 0 events as they have some weird pileup.
centrality = [[6., 14., 28., 50., 84., 131., 197., 285., 408., 492.],
              [3., 7., 15., 27., 45., 72., 108., 157., 227., 274.],
              [3.19, 4.51, 6.38, 7.8, 9.01, 10.07, 11.03, 11.92, 12.75, 13.61],
              [4.338, 13.958, 29.858, 55.32, 92.322, 142.58, 207.161, 289.727, 394.215, 457.208],
              [6.503, 13.531, 27.266, 49.652, 83.318, 131.64, 196.276, 283.764, 404.833, 485.149],
              [6.076, 13.573, 27.286, 49.105, 83.39, 131.38, 195.412, 283.929, 403.789, 484.684],
              [4.991, 12.16, 25.672, 46.537, 79.563, 126.518, 192.391, 281.273, 401.452, 481.841]]
# CBWC arrays.
C_cbwc = np.zeros((4, 3, len(ylabels), len(centrality[0]) + 1))
E_cbwc = np.zeros((4, 3, len(ylabels), len(centrality[0]) + 1))
C_rat_cbwc = np.zeros((4, 3, len(ylabels), len(centrality[0]) + 1))
E_rat_min_cbwc = np.zeros((4, 3, len(ylabels), len(centrality[0]) + 1))
E_rat_max_cbwc = np.zeros((4, 3, len(ylabels), len(centrality[0]) + 1))
for k in range(4):
    for i in range(3):
        for j in range(len(ylabels)):
            for r in range(len(centrality[0]) + 1):
                if r == 0:
                    index = p_bins[k][i][j] <= centrality[j][r]
                    C_cbwc[k][i][j][r] = np.divide(np.sum(np.multiply(n_bins[k][i][j][index], C[k][i][j][index])),
                                                   np.sum(n_bins[k][i][j][index]))
                    E_cbwc[k][i][j][r] = np.divide(np.sum(np.multiply(n_bins[k][i][j][index], E[k][i][j][index])),
                                                   np.sum(n_bins[k][i][j][index]))
                    C_rat_cbwc[k][i][j][r] = np.divide(np.sum(np.multiply(n_bins[k][i][j][index],
                                                                          C_rat[k][i][j][index])),
                                                       np.sum(n_bins[k][i][j][index]))
                    E_rat_min_cbwc[k][i][j][r] = np.divide(np.sum(np.multiply(n_bins[k][i][j][index],
                                                                              E_rat_min[k][i][j][index])),
                                                           np.sum(n_bins[k][i][j][index]))
                    E_rat_max_cbwc[k][i][j][r] = np.divide(np.sum(np.multiply(n_bins[k][i][j][index],
                                                                              E_rat_max[k][i][j][index])),
                                                           np.sum(n_bins[k][i][j][index]))
                elif r < len(centrality[0]) - 1:
                    index = (p_bins[k][i][j] <= centrality[j][r]) & (p_bins[k][i][j] > centrality[j][r-1])
                    C_cbwc[k][i][j][r] = np.divide(np.sum(np.multiply(n_bins[k][i][j][index], C[k][i][j][index])),
                                                   np.sum(n_bins[k][i][j][index]))
                    E_cbwc[k][i][j][r] = np.divide(np.sum(np.multiply(n_bins[k][i][j][index], E[k][i][j][index])),
                                                   np.sum(n_bins[k][i][j][index]))
                    C_rat_cbwc[k][i][j][r] = np.divide(
                        np.sum(np.multiply(n_bins[k][i][j][index], C_rat[k][i][j][index])),
                        np.sum(n_bins[k][i][j][index]))
                    E_rat_min_cbwc[k][i][j][r] = np.divide(np.sum(np.multiply(n_bins[k][i][j][index],
                                                                              E_rat_min[k][i][j][index])),
                                                           np.sum(n_bins[k][i][j][index]))
                    E_rat_max_cbwc[k][i][j][r] = np.divide(np.sum(np.multiply(n_bins[k][i][j][index],
                                                                              E_rat_max[k][i][j][index])),
                                                           np.sum(n_bins[k][i][j][index]))
                else:
                    index = p_bins[k][i][j] > centrality[j][r-1]
                    C_cbwc[k][i][j][r] = np.divide(np.sum(np.multiply(n_bins[k][i][j][index], C[k][i][j][index])),
                                                   np.sum(n_bins[k][i][j][index]))
                    E_cbwc[k][i][j][r] = np.divide(np.sum(np.multiply(n_bins[k][i][j][index], E[k][i][j][index])),
                                                   np.sum(n_bins[k][i][j][index]))
                    C_rat_cbwc[k][i][j][r] = np.divide(
                        np.sum(np.multiply(n_bins[k][i][j][index], C_rat[k][i][j][index])),
                        np.sum(n_bins[k][i][j][index]))
                    E_rat_min_cbwc[k][i][j][r] = np.divide(np.sum(np.multiply(n_bins[k][i][j][index],
                                                                              E_rat_min[k][i][j][index])),
                                                           np.sum(n_bins[k][i][j][index]))
                    E_rat_max_cbwc[k][i][j][r] = np.divide(np.sum(np.multiply(n_bins[k][i][j][index],
                                                                              E_rat_max[k][i][j][index])),
                                                           np.sum(n_bins[k][i][j][index]))

print("RM3")
print(C_rat_cbwc[2][2][0])
print(E_rat_min_cbwc[2][2][0])
print(E_rat_max_cbwc[2][2][0])
print("b")
print(C_rat_cbwc[2][2][2])
print(E_rat_min_cbwc[2][2][2])
print(E_rat_max_cbwc[2][2][2])
print("ReLU")
print(C_rat_cbwc[2][2][4])
print(E_rat_min_cbwc[2][2][4])
print(E_rat_max_cbwc[2][2][4])
plt.plot(1)
plt.show()

C_labels_cbwc = [r'$C_1$', r'$C_2$', r'$C_3$', r'$C_4$']
for i in range(4):
    fig, ax = plt.subplots(3, len(ylabels), figsize=(16, 9), constrained_layout=True)
    for j in range(3):
        for k in range(len(ylabels)):
            r = j * len(ylabels)
            if k == 2:
                ax[j][k].fill_between(centralities, C_cbwc[i][j][k][::-1] + E_cbwc[i][j][k][::-1],
                                      C_cbwc[i][j][k][::-1] - E_cbwc[i][j][k][::-1],
                                      color=color[k], alpha=0.5)
                ax[j][k].plot(centralities, C_cbwc[i][j][k][::-1], marker=markers[k], ms=10, lw=0, mfc='none',
                              color=color[k])
            else:
                ax[j][k].fill_between(centralities, C_cbwc[i][j][k] + E_cbwc[i][j][k],
                                      C_cbwc[i][j][k] - E_cbwc[i][j][k],
                                      color=color[k], alpha=0.5)
                ax[j][k].plot(centralities, C_cbwc[i][j][k], marker=markers[k], ms=10, lw=0, mfc='none',
                              color=color[k])
            ax[j][k].set_xticklabels(centralities, rotation=90)
            ax[j][k].set_xlabel(ylabels[k], fontsize=15)
    fig.suptitle(C_labels_cbwc[i], fontsize=20)
    # plt.show()
    plt.close()
for i in range(4):
    fig, ax = plt.subplots(3, len(ylabels), figsize=(16, 9), constrained_layout=True)
    for j in range(3):
        for k in range(len(ylabels)):
            r = j * len(ylabels)
            if k == 2:
                ax[j][k].fill_between(centralities, E_rat_max_cbwc[i][j][k][::-1], E_rat_min_cbwc[i][j][k][::-1],
                                      color=color[k], alpha=0.5)
                ax[j][k].plot(centralities, C_rat_cbwc[i][j][k][::-1], color=color[k], marker=markers[k], ms=10, lw=0,
                              mfc='none')
            else:
                ax[j][k].fill_between(centralities, E_rat_max_cbwc[i][j][k], E_rat_min_cbwc[i][j][k],
                                      color=color[k], alpha=0.5)
                ax[j][k].plot(centralities, C_rat_cbwc[i][j][k], color=color[k], marker=markers[k], ms=10, lw=0,
                              mfc='none')
            ax[j][k].set_xticklabels(centralities, rotation=90)
            ax[j][k].set_xlabel(ylabels[k], fontsize=15)
    fig.suptitle(C_labels_rat[i], fontsize=20)
    # plt.show()
    plt.close()
for i in range(4):
    plt.figure(figsize=(12, 8))
    for k in range(len(EPD_set)):
        if EPD_set[k] == 2:
            plt.fill_between(centralities, C_cbwc[i][2][EPD_set[k]][::-1] + E_cbwc[i][2][EPD_set[k]][::-1],
                                  C_cbwc[i][2][EPD_set[k]][::-1] - E_cbwc[i][2][EPD_set[k]][::-1],
                                  color=color[EPD_set[k]], alpha=0.5)
            plt.plot(centralities, C_cbwc[i][j][EPD_set[k]][::-1], marker=markers[EPD_set[k]], ms=10, lw=0, mfc='none',
                          color=color[EPD_set[k]], label=ylabels[EPD_set[k]])
        else:
            plt.fill_between(centralities, C_cbwc[i][j][EPD_set[k]] + E_cbwc[i][j][EPD_set[k]],
                                  C_cbwc[i][j][EPD_set[k]] - E_cbwc[i][j][EPD_set[k]],
                                  color=color[EPD_set[k]], alpha=0.5)
            plt.plot(centralities, C_cbwc[i][j][EPD_set[k]], marker=markers[EPD_set[k]], ms=10, lw=0, mfc='none',
                          color=color[EPD_set[k]], label=ylabels[EPD_set[k]])
        plt.xticks(rotation=90)
        plt.xlabel('X', fontsize=15)
    plt.title(C_labels_cbwc[i], fontsize=20)
    plt.legend()
    # plt.show()
    plt.close()
for i in range(4):
    plt.figure(figsize=(12, 8))
    for k in range(len(EPD_set)):
        if EPD_set[k] == 2:
            plt.fill_between(centralities, E_rat_max_cbwc[i][2][EPD_set[k]][::-1],
                                  E_rat_min_cbwc[i][2][EPD_set[k]][::-1],
                                  color=color[EPD_set[k]], alpha=0.5)
            plt.plot(centralities, C_rat_cbwc[i][j][EPD_set[k]][::-1], marker=markers[EPD_set[k]], ms=10, lw=0,
                     mfc='none', color=color[EPD_set[k]], label=ylabels[EPD_set[k]])
        else:
            plt.fill_between(centralities, E_rat_max_cbwc[i][j][EPD_set[k]],
                                  E_rat_min_cbwc[i][j][EPD_set[k]],
                                  color=color[EPD_set[k]], alpha=0.5)
            plt.plot(centralities, C_rat_cbwc[i][j][EPD_set[k]], marker=markers[EPD_set[k]], ms=10, lw=0, mfc='none',
                          color=color[EPD_set[k]], label=ylabels[EPD_set[k]])
        plt.xticks(rotation=90)
        plt.xlabel('X', fontsize=15)
    plt.title(C_labels[i], fontsize=20)
    plt.legend()
    # plt.show()
    plt.close()
for i in range(4):
    plt.figure(figsize=(12, 8))
    for k in range(len(Final_set)):
        if Final_set[k] == 2:
            plt.fill_between(centralities, C_cbwc[i][2][Final_set[k]][::-1] + E_cbwc[i][2][Final_set[k]][::-1],
                             C_cbwc[i][2][Final_set[k]][::-1] - E_cbwc[i][2][Final_set[k]][::-1],
                             color=color[Final_set[k]], alpha=0.5)
            plt.plot(centralities, C_cbwc[i][j][Final_set[k]][::-1], marker=markers[Final_set[k]], ms=10, lw=0,
                     mfc='none', color=color[Final_set[k]], label=ylabels[Final_set[k]])
        else:
            plt.fill_between(centralities, C_cbwc[i][j][Final_set[k]] + E_cbwc[i][j][Final_set[k]],
                             C_cbwc[i][j][Final_set[k]] - E_cbwc[i][j][Final_set[k]],
                             color=color[Final_set[k]], alpha=0.5)
            plt.plot(centralities, C_cbwc[i][j][Final_set[k]], marker=markers[Final_set[k]], ms=10, lw=0, mfc='none',
                     color=color[Final_set[k]], label=ylabels[Final_set[k]])
        plt.xticks(rotation=90)
        plt.xlabel('X', fontsize=15)
    plt.title(C_labels_cbwc[i], fontsize=20)
    plt.legend()
    plt.show()
    plt.close()
for i in range(4):
    plt.figure(figsize=(12, 8))
    for k in range(len(Final_set)):
        if Final_set[k] == 2:
            plt.fill_between(centralities, E_rat_max_cbwc[i][2][Final_set[k]][::-1],
                             E_rat_min_cbwc[i][2][Final_set[k]][::-1], color=color[Final_set[k]], alpha=0.5)
            plt.plot(centralities, C_rat_cbwc[i][j][Final_set[k]][::-1], marker=markers[Final_set[k]], ms=10, lw=0,
                     mfc='none', color=color[Final_set[k]], label=ylabels[Final_set[k]])
        else:
            plt.fill_between(centralities, E_rat_max_cbwc[i][j][Final_set[k]],
                             E_rat_min_cbwc[i][j][Final_set[k]], color=color[Final_set[k]], alpha=0.5)
            plt.plot(centralities, C_rat_cbwc[i][j][Final_set[k]], marker=markers[Final_set[k]], ms=10, lw=0,
                     mfc='none', color=color[Final_set[k]], label=ylabels[Final_set[k]])
        plt.xticks(rotation=90)
        plt.xlabel('X', fontsize=15)
    plt.title(C_labels[i], fontsize=20)
    plt.legend()
    plt.show()
    plt.close()
#######################################################################################################
# This is a sad bunch of noise right now; ignore it. If I fix it, you won't even see this message ... #
#######################################################################################################
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
            C_cbwc[k][i].append(rdr.cbwc(pro_counts[i][j],  # TODO Rewrite CBWC with new array values.
                                         pro_bins[i][j],  # TODO Don't forget you made n_bins!
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
plt_set = (0, 1, 2, 3, 4)
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
            print("Values for " + ylabels[j] + ":")
            print(centralities)
            print(fill_up[1][-1][j][::-1])
            print(fill_down[1][-1][j][::-1])
        else:
            plt.fill_between(centralities, fill_up[1][-1][j], fill_down[1][-1][j], alpha=0.2,
                             color=color[j])
            plt.scatter(centralities, C_cbwc[1][-1][j],
                        marker=markers[j], s=size, alpha=alpha, c=color[j],
                        label=ylabels[j])
            print("Values for " + ylabels[j] + ":")
            print(centralities)
            print(fill_up[1][-1][j])
            print(fill_down[1][-1][j])
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
