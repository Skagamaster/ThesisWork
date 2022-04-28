# \author Skipper Kagamaster
# \date 03/18/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
#

"""
This code is made to display the correlations between
centrality metrics and net-protons moments for STAR
FastOffline data.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import reader as rdr
from scipy.stats import moment

ylabels = [r'$X_{RM3}$', r'$X_{LW}$', r'$X_{ReLU}$', r'$X_{swish}$', r'$X_{CNN}$']
ylabels = [r'RefMult3', r'$EPD_{LW}$', r'$EPD_{ReLU}$', r'$EPD_{swish}$', r'$EPD_{CNN}$']
xlabels = ['protons', 'antiprotons', 'net protons']
# Let's load up the data.
data_directory = r'C:\Users\dansk\Documents\Thesis\Protons\2022_data'
os.chdir(data_directory)
# This is just RefMult3
refmult = np.load('refmult.npy', allow_pickle=True)
# Array of [protons, antiprotons, net_protons]
protons = np.load('protons.npy', allow_pickle=True)
# Array of ML predictions: [LW, ReLU, Swish, CNN]
epd_ML = np.load('predictions_14_refmult3.npy', allow_pickle=True).astype('int')
X_cent = np.vstack((refmult, epd_ML))
print("Data loaded. Starting analysis.")

# Let's use quantiles to get our centrality cuts.
cent_bins = np.hstack((np.linspace(10, 90, 9), (95, 97, 99)))
centralities = ['90-100%', '80-90%', '70-80%', '60-70%', '50-60%', '40-50%',
                '30-40%', '20-30%', '10-20%', '5-10%', '3-5%', '1-3%', '0-1%']
cent_bins = np.hstack((np.linspace(10, 90, 9), 95))
centralities = ['90-100%', '80-90%', '70-80%', '60-70%', '50-60%', '40-50%',
                '30-40%', '20-30%', '10-20%', '5-10%', '0-5%']
cent_bins = np.hstack((np.linspace(10, 90, 9), (95, 97, 99)))
centralities = ['90-100%', '80-90%', '70-80%', '60-70%', '50-60%', '40-50%',
                '30-40%', '20-30%', '10-20%', '5-10%', '3-5%', '1-3%', '0-1%']
centrality = []
print("Quantile centralities:")
print(centralities)
count = 0
for i in X_cent:
    print(ylabels[count])
    arr = np.percentile(i, cent_bins).astype('float')
    centrality.append(arr)
    print(arr)
    count += 1
centrality = np.asarray(centrality)
np.save("centrality.npy", centrality)

"""
The data gets pretty messy for peripheral collisions,
so we omit RefMult3 = 0 from the moments.
"""
index_0 = (refmult > 0)
refmult = refmult[index_0]
X_cent_ = []
for i in range(len(X_cent)):
    X_cent_.append([])
    X_cent_[i] = X_cent[i][index_0]
protons_ = []
for i in range(len(protons)):
    protons_.append([])
    protons_[i] = protons[i][index_0]
X_cent = X_cent_
protons = protons_

"""
This can be made less computationally expensive by combining
with the above loop, but it's pretty quick as is.
"""
print("Finding moments.")
x = []
n_cent = []
C = []
C_labels = [r'$C_1$', r'$C_2$', r'$C_3$', r'$C_4$']
E = []
E_rat = []
"""
C and E are structured as:
C[X_cent][pro/anti/net][C1/C2/C3/C4]
"""
for i in range(len(X_cent)):
    C.append([])
    E.append([])
    E_rat.append([])
    for j in range(3):
        C[i].append([])
        E[i].append([])
        E_rat[i].append([])
        for k in range(4):
            C[i][j].append([])
            E[i][j].append([])
            E_rat[i][j].append([])
for i in range(len(X_cent)):
    print("Working on {}.".format(ylabels[i]))
    x.append(np.sort(np.unique(X_cent[i])))
    n_cent.append([])
    # TODO: Find a way to broadcast this? I don't think it's possible.
    for j in x[i]:
        index = (X_cent[i] == j)
        pro_arr = protons[0][index]
        apro_arr = protons[1][index]
        npro_arr = protons[2][index]
        n = len(pro_arr)
        n_cent[i].append(n)
        # We need moments 1-8 for each distribution.
        u = []
        au = []
        nu = []
        u.append(np.mean(pro_arr))
        au.append(np.mean(apro_arr))
        nu.append(np.mean(npro_arr))
        for k in range(1, 8):
            u.append(moment(pro_arr, k+1))
            au.append(moment(apro_arr, k+1))
            nu.append(moment(npro_arr, k+1))
        for k in range(4):
            C[i][0][k].append(rdr.cn_data(u, k))
            C[i][1][k].append(rdr.cn_data(au, k))
            C[i][2][k].append(rdr.cn_data(nu, k))
            E[i][0][k].append(rdr.en_data(u, n, k))
            E[i][1][k].append(rdr.en_data(au, n, k))
            E[i][2][k].append(rdr.en_data(nu, n, k))
            E_rat[i][0][k].append(rdr.ern_data(u, n, k))
            E_rat[i][1][k].append(rdr.ern_data(au, n, k))
            E_rat[i][2][k].append(rdr.ern_data(nu, n, k))
x = np.asarray(x, dtype='object')
n_cent = np.asarray(n_cent, dtype='object')
C = np.asarray(C, dtype='object')
E = np.asarray(E, dtype='object')
E_rat = np.asarray(E_rat, dtype='object')
np.save("x.npy", x)
np.save('n_cent.npy', n_cent)
np.save("C.npy", C)
np.save("E.npy", E)
np.save("E_rat.npy", E_rat)

# rdr.cum_plot_int_data(ylabels, x, C, C_labels, xlabels, gev='14.6')
# rdr.cum_plot_int_err_data(ylabels, x, C, C_labels, E, xlabels, gev='14.6')

""" Now for ratios and CBWC. """
C_rat = np.copy(C)
for i in range(len(X_cent)):
    for j in range(3):
        C_rat[i][j][2] = np.divide(C[i][j][2], C[i][j][1])
        C_rat[i][j][3] = np.divide(C[i][j][3], C[i][j][1])
        C_rat[i][j][1] = np.divide(C[i][j][1], C[i][j][0])
np.save("C_rat.npy", C_rat)
C_rat_labels = [r'$\mu$', r'$\frac{\sigma^2}{\mu}$',
                r'$S\sigma$', r'$\kappa\sigma^2$']
# rdr.cum_plot_int_data(ylabels, x, C_rat, C_rat_labels, xlabels, gev='14.6')
# rdr.cum_plot_int_err_data(ylabels, x, C_rat, C_rat_labels, E_rat, xlabels, gev='14.6')

"""
Now to get the CBWC results for the ratios.
"""
C_cbwc = []
E_cbwc = []
for i in range(len(X_cent)):
    C_cbwc.append([])
    E_cbwc.append([])
    for j in range(3):
        C_cbwc[i].append([])
        E_cbwc[i].append([])
        for k in range(4):
            C_cbwc[i][j].append([])
            E_cbwc[i][j].append([])
for i in range(len(X_cent)):
    for j in range(3):
        for k in range(4):
            C_cbwc[i][j][k] = rdr.cbwc_data(C_rat[i][j][k], n_cent[i], x[i], centrality[i])
            E_cbwc[i][j][k] = rdr.cbwc_data(E_rat[i][j][k], n_cent[i], x[i], centrality[i])
C_cbwc = np.asarray(C_cbwc, dtype='object')
E_cbwc = np.asarray(E_cbwc, dtype='object')
np.save("C_cbwc.npy", C_cbwc)
np.save("E_cbwc.npy", E_cbwc)
# rdr.cum_plot_int_err_data(ylabels, centralities, C_cbwc, C_rat_labels, E_cbwc, xlabels)

plot_cbwc = []
for i in range(len(X_cent)):
    plot_cbwc.append([])
    for j in range(2):
        if j == 0:
            plot_cbwc[i].append(C_cbwc[i][2][1] + E_cbwc[i][2][1])
        else:
            plot_cbwc[i].append(C_cbwc[i][2][1] - E_cbwc[i][2][1])
plot_cbwc = np.asarray(plot_cbwc, dtype='float')

markers = ['o', '8', 's', 'P', '*', 'X', 'D']
color = ['orangered', 'orange', 'black', 'blue', 'purple', 'darkviolet', 'deepskyblue']

# This is the end, final plot. Are there ACEs shown?
plt.figure(figsize=(9, 5))
# plt.title(r'$\frac{\sigma^2}{\mu}$ for $\sqrt{s_{NN}}$ = 14.6 GeV (STAR Data)', fontsize=30)
for j in range(len(X_cent)):
    plt.fill_between(centralities, plot_cbwc[j][0], plot_cbwc[j][1], color=color[j],
                     alpha=0.2)
    plt.scatter(centralities, C_cbwc[j][-1][1], marker=markers[j], s=50, alpha=0.5, c=color[j],
                label=ylabels[j])
plt.xticks(centralities, labels=centralities, rotation=45)
plt.xlabel("Centrality", fontsize=20)
plt.ylabel(r'$\frac{\sigma^2}{\mu}$', fontsize=20)
plt.legend()
plt.tight_layout()
plt.show()

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

plt.figure(figsize=(9, 5))
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
plt.legend()
plt.tight_layout()
plt.show()

# And with error (such that it is, and only some values).
plt.figure(figsize=(16, 9))
plt.title(r'$\mu$ for $\sqrt{s_{NN}}$=19.6 GeV, UrQMD', fontsize=30)
for j in plt_set:
    if target == 'b':
        if j > 1:
            plt.errorbar(centralities, C_cbwc[0][-1][j][::-1],
                         E_cbwc[0][-1][j][::-1],
                         marker=markers[j], ms=10, alpha=0.5, c=color[j],
                         label=ylabels[j], lw=0, elinewidth=1, capsize=2)
        else:
            plt.errorbar(centralities, C_cbwc[0][-1][j],
                         E_cbwc[0][-1][j],
                         marker=markers[j], ms=10, alpha=0.5, c=color[j],
                         label=ylabels[j], lw=0, elinewidth=1, capsize=2)
    else:
        if j == 2:
            plt.errorbar(centralities, C_cbwc[0][-1][j][::-1],
                         E_cbwc[0][-1][j][::-1],
                         marker=markers[j], ms=10, alpha=0.5, c=color[j],
                         label=ylabels[j], lw=0, elinewidth=1, capsize=2)
        else:
            plt.errorbar(centralities, C_cbwc[0][-1][j],
                         E_cbwc[0][-1][j],
                         marker=markers[j], ms=10, alpha=0.5, c=color[j],
                         label=ylabels[j], lw=0, elinewidth=1, capsize=2)
plt.xticks(centralities, labels=centralities, rotation=45)
plt.xlabel("Centrality", fontsize=20)
plt.ylabel(r'$\mu$', fontsize=20)
plt.legend()
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
