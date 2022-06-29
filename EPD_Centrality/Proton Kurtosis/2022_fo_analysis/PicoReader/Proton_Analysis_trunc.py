#
# \Analyse truncated proton arrays.
#
# \author Skipper Kagamaster
# \date 06/09/2022
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

df = pd.read_pickle(r'D:\14GeV\Thesis\Proton_Analysis_WIP\ML\trunc_set.pkl')
# ReLU and Linear sets need to be made into integer values.
df['linear'] = df['linear'].astype('int')
df['relu'] = df['relu'].astype('int')
# And now let's make just the raw, EPD nMIP sum.
epd_sum = df['ring1']
for i in range(1, 32):
    epd_sum += df['ring{}'.format(i+1)]
df['epd_sum'] = epd_sum.astype('int')

# Let's do a plot of the centrality measures (X).
plt.hist(df['epd_sum'], bins=750, range=(0, 750), histtype='step', density=True, label=r'$X_{\Sigma}$',
         color='g', alpha=0.5, lw=2)
plt.hist(df['relu'], bins=750, range=(0, 750), histtype='step', density=True, label=r'$X_{ReLU}$',
         color='b', alpha=0.5, lw=2)
plt.hist(df['RefMult3'], bins=750, range=(0, 750), histtype='step', density=True, label=r'$X_{RM3}$',
         color='orange', alpha=0.5, lw=2)
plt.xlabel(r"$X$", fontsize=15, loc='right')
plt.ylabel(r"$\frac{dN}{dX}$", fontsize=15, loc='top')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

# Now for some comparison plots.
fig, ax = plt.subplots(1, 2, figsize=(12, 7), constrained_layout=True)
count1, binsX1, binsY1 = np.histogram2d(df['RefMult3'], df['linear'], bins=(700, 800),
                                        range=((0, 700), (-100, 700)))
count2, binsX2, binsY2 = np.histogram2d(df['RefMult3'], df['relu'], bins=700, range=((0, 700), (0, 700)))
count3, binsX3, binsY3 = np.histogram2d(df['RefMult3'], df['epd_sum'], bins=700, range=((0, 700), (0, 700)))
X, Y = np.meshgrid(binsX3, binsY3)
im1 = ax[0].pcolormesh(X, Y, count3.T, cmap='jet', norm=LogNorm())
ax[0].set_xlabel(r"$X_{RM3}$", fontsize=15)
ax[0].set_ylabel(r'$X_{\Sigma}$', fontsize=15)
fig.colorbar(im1, ax=ax[0])
X, Y = np.meshgrid(binsX2, binsY2)
im2 = ax[1].pcolormesh(X, Y, count2.T, cmap='jet', norm=LogNorm())
ax[1].set_xlabel(r"$X_{RM3}$", fontsize=15)
ax[1].set_ylabel(r'$X_{ReLU}$', fontsize=15)
fig.colorbar(im1, ax=ax[1])
plt.show()
plt.close()

# Now to do a cumulant analysis for all X byb integer value.
c = []
e = []
c_r = []
e_r = []
x_vals = []
n_vals = []
for i in range(3):
    c.append([[], [], [], []])
    e.append([[], [], [], []])
    c_r.append([[], [], [], []])
    e_r.append([[], [], [], []])
    n_vals.append([])
x_set = ['RefMult3', 'epd_sum', 'relu']
for i in range(3):
    print("This is", x_set[i] + '.')
    vals = np.unique(df[x_set[i]])
    x_vals.append(vals)
    for m in vals:
        arr = df['net_protons'][df[x_set[i]] == m]
        n_vals[i].append(len(arr))
        n = np.sqrt(len(arr))
        u = pr.moment_arr(arr)
        c[i][0].append(u[0])
        c[i][1].append(u[1])
        c[i][2].append(u[2])
        c[i][3].append(u[3] - 3 * (u[1] ** 2))

        err = pr.err(arr, u)
        e[i][0].append(np.sqrt(err[0]) / n)
        e[i][1].append(np.sqrt(err[1]) / n)
        e[i][2].append(np.sqrt(err[2]) / n)
        e[i][3].append(np.sqrt(err[3]) / n)

        c_r[i][0].append(u[0])
        c_r[i][1].append(u[1]/np.max((u[0], 1e-6)))
        c_r[i][2].append(u[2]/np.max((u[1], 1e-6)))
        c_r[i][3].append((u[3] - 3 * (u[1] ** 2))/np.max((u[1], 1e-6)))

        err_rat = pr.err_rat(arr, u)
        e_r[i][0].append(np.sqrt(err[0]) / n)
        e_r[i][1].append(err_rat[0] / n)
        e_r[i][2].append(err_rat[1]/n)
        e_r[i][3].append(err_rat[2]/n)
df_rm3 = pd.DataFrame(c[0][0])
df_rm3.columns = ['c1']
df_rm3.index = x_vals[0]
df_lin = pd.DataFrame(c[1][0])
df_lin.columns = ['c1']
df_lin.index = x_vals[1]
df_relu = pd.DataFrame(c[2][0])
df_relu.columns = ['c1']
df_relu.index = x_vals[2]
for i in range(1, 5):
    if i > 1:
        df_rm3['c{}'.format(i)] = c[0][i - 1]
        df_lin['c{}'.format(i)] = c[1][i - 1]
        df_relu['c{}'.format(i)] = c[2][i - 1]
    df_rm3['cr{}'.format(i)] = c_r[0][i - 1]
    df_rm3['er{}'.format(i)] = e_r[0][i - 1]
    df_lin['cr{}'.format(i)] = c_r[1][i - 1]
    df_lin['er{}'.format(i)] = e_r[1][i - 1]
    df_relu['cr{}'.format(i)] = c_r[2][i - 1]
    df_relu['er{}'.format(i)] = e_r[2][i - 1]
    df_rm3['e{}'.format(i)] = e[0][i-1]
    df_lin['e{}'.format(i)] = e[1][i-1]
    df_relu['e{}'.format(i)] = e[2][i-1]
df_rm3['n'] = n_vals[0]
df_lin['n'] = n_vals[1]
df_relu['n'] = n_vals[2]
df_rm3.to_pickle(r'D:\14GeV\Thesis\Proton_Analysis_WIP\ML\rm3_cumulants.pkl')
df_lin.to_pickle(r'D:\14GeV\Thesis\Proton_Analysis_WIP\ML\lin_cumulants.pkl')
df_relu.to_pickle(r'D:\14GeV\Thesis\Proton_Analysis_WIP\ML\relu_cumulants.pkl')
print("All pickled.")

# Now some plots of the cumulants of net-protons by RefMult3 integer, for each df.
t_labels = [r'$\mu$', r'$\sigma^2$', r'$S$', r'$\kappa$']
x_labels = [r'$X_{RM3}$', r'$X_{\Sigma}$', r'$X_{ReLU}$']
ranges = ((0, 40), (0, 60), (-100, 100), (-100, 100))
text_high = (35, 55, 95, 95)
# No error
fig, ax = plt.subplots(3, 4, figsize=(16, 9), constrained_layout=True)
for i in range(3):
    for j in range(4):
        ax[i, j].scatter(x_vals[i], c[i][j],
                         marker='o', c='k', s=10)
        ax[i, j].set_xlabel(x_labels[i], fontsize=12, loc='right')
        ax[i, j].set_ylabel(t_labels[j], fontsize=15)
        ax[i, j].set_ylim(ranges[j])
plt.show()
plt.close()
# With error
fig, ax = plt.subplots(3, 4, figsize=(16, 9), constrained_layout=True)
for i in range(3):
    for j in range(4):
        ax[i, j].errorbar(x_vals[i], c[i][j], yerr=e[i][j],
                          marker='o', color='k', ms=2,
                          mfc='None', elinewidth=0.1, lw=0,
                          ecolor='black')
        ax[i, j].set_xlabel(x_labels[i], fontsize=12, loc='right')
        ax[i, j].set_ylabel(t_labels[j], fontsize=15)
        ax[i, j].set_ylim(ranges[j])
plt.show()
plt.close()
# Now again, but for the ratios.
t_labels = [r'$\mu$', r'$\frac{\sigma^2}{\mu}$', r'$S\sigma$', r'$\kappa\sigma^2$']
ranges = ((0, 40), (0.5, 3.5), (-100, 100), (-1000, 1000))
text_high = (3.0, 95, 95)
# No error
fig, ax = plt.subplots(3, 4, figsize=(16, 9), constrained_layout=True)
for i in range(3):
    for j in range(4):
        ax[i, j].scatter(x_vals[i], c_r[i][j],
                         marker='o', c='k', s=10)
        ax[i, j].set_xlabel(x_labels[i], fontsize=12, loc='right')
        ax[i, j].set_ylabel(t_labels[j], fontsize=15)
        ax[i, j].set_ylim(ranges[j])
plt.show()
plt.close()
# With error
fig, ax = plt.subplots(3, 4, figsize=(16, 9), constrained_layout=True)
for i in range(3):
    for j in range(4):
        ax[i, j].errorbar(x_vals[i], c_r[i][j], yerr=e_r[i][j],
                          marker='o', color='k', ms=2,
                          mfc='None', elinewidth=0.1, lw=0,
                          ecolor='black')
        ax[i, j].set_xlabel(x_labels[i], fontsize=12, loc='right')
        ax[i, j].set_ylabel(t_labels[j], fontsize=15)
        ax[i, j].set_ylim(ranges[j])
plt.show()
plt.close()
