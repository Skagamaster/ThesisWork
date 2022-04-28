import numpy as np
import pandas as pd
import uproot as up
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

os.chdir(r'C:\Users\dansk\Documents\Thesis\Protons\2022_urqmd\pro_count_archives\1M_events_refmult3_0.4_2.0')

# First, let's find centrality and centrality resolutions.
ring = np.load('ring.npy', allow_pickle=True)  # RM3, RM1, b, EPD rings
predictions = np.load('predictions_14_refmult3.npy', allow_pickle=True)  # ML predictions (ReLU, Swish, CNN)
epd = np.sum(ring[4:], axis=0)  # Build EPD sum
epd_rings = np.add(ring[3:19], ring[19:])
refmults = np.vstack((ring[:3], predictions))
refmults = np.vstack((refmults, epd))
index = refmults[1] != 0  # There are a lot of empty events at RM3 == 0; omit from analysis
refmults = refmults.T[index].T
epd_rings = epd_rings.T[index].T

# Plotting EPD rings vs RM3.
fig, ax = plt.subplots(4, 4, figsize=(16, 9), constrained_layout=True)
for i in range(4):
    for j in range(4):
        x = i * 4 + j
        bins = 35
        ranger = (0, 35)
        if x == 0:
            bins = 110
            ranger = (0, 110)
        if x == 1:
            bins = 70
            ranger = (0, 70)
        count, binsX, binsY = np.histogram2d(epd_rings[x], refmults[1], bins=(bins, 700), range=(ranger, (0, 700)))
        count[count <= 5] = 0
        X, Y = np.meshgrid(binsY, binsX)
        im = ax[i, j].pcolormesh(X, Y, count, norm=LogNorm(), cmap='jet')
        fig.colorbar(im, ax=ax[i, j], aspect=10)
        ax[i, j].set_title('Ring {}'.format(x + 1), fontsize=15)
# plt.show()
plt.close()

# Let's plot the multiplicity(ish) distributions.
labels = ['b', r'$X_{RM3}$', r'$X_{RM1}$', r'$X_{LW}$', r'$X_{ReLU}$', r'$X_{Swish}$', r'$X_{CNN}$',
          r'$X_{\Sigma}$']
markers = ['o', '8', 's', 'P', '*', 'X', 'D']
color = ['orangered', 'orange', 'black', 'blue', 'purple', 'darkviolet', 'royalblue']
fig, ax = plt.subplots(1, figsize=(12, 7), constrained_layout=True)
for i in range(len(refmults) - 1):
    ax.hist(refmults[i + 1], bins=750, range=(-50, 700), histtype='step', label=labels[i + 1], lw=2, alpha=0.7,
            density=True, color=color[i])
plt.legend(fontsize=15)
plt.yscale('log')
plt.xlabel("X", fontsize=20, loc='right')
plt.ylabel(r'$\frac{dN}{dX}$', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.show()
plt.close()

# Now we'll find centrality based on quantile cuts.
centrality = []
cent_bins = np.hstack((np.linspace(10, 90, 9), 95))
cent_bins_reverse = np.hstack((5, np.linspace(10, 90, 9)))
x_cent = ["70-80%", "60-70%", "50-60%", "40-50%", "30-40%",
          "20-30%", "10-20%", "5-10%", "0-5%"]
centrality.append(np.percentile(refmults[0], cent_bins_reverse)[::-1])
for i in range(len(refmults) - 1):
    centrality.append(np.percentile(refmults[i + 1], cent_bins))
centrality = np.around(np.asarray(centrality, dtype='float'), 3)
print(centrality)

# Now to calculate centrality resolutions. We first get the var for b
# distributions based on quantiles for b, then for b distributions
# based on quantiles for the other measures. Resolution is:
# phi = var(others)/var(b)
phi_b = []
index = refmults[0] >= centrality[0][0]
phi_b.append(np.var(refmults[0][index]))
for i in range(len(centrality[0]) - 1):
    index = (refmults[0] >= centrality[0][i + 1]) & (refmults[0] < centrality[0][i])
    phi_b.append(np.var(refmults[0][index]))
index = refmults[0] < centrality[0][-1]
phi_b.append(np.var(refmults[0][index]))
phi_b = np.asarray(phi_b, dtype='float')

phi = []
dist_b = []
for i in range(len(refmults) - 1):
    phi.append([])
    dist_b.append([])
    index = refmults[i + 1] <= centrality[i + 1][0]
    phi[i].append(np.var(refmults[0][index]))
    dist_b[i].append(refmults[0][index])
    for j in range(len(centrality[0]) - 1):
        index = (refmults[i + 1] <= centrality[i + 1][j + 1]) & (refmults[i + 1] > centrality[i + 1][j])
        phi[i].append(np.var(refmults[0][index]))
        dist_b[i].append(refmults[0][index])
    index = refmults[i + 1] > centrality[i + 1][-1]
    phi[i].append(np.var(refmults[0][index]))
    dist_b[i].append(refmults[0][index])
phi = np.asarray(phi, dtype='float')
for i in range(len(phi)):
    phi[i] = np.divide(phi[i], phi_b)

pass

fig, ax = plt.subplots(1, figsize=(12, 7), constrained_layout=True)
subset = [2, 3, 4, 5, 6]
for i in subset:
    ax.plot(phi[i][2:], label=labels[i + 1], color=color[i], marker=markers[i], ms=15)
plt.legend(fontsize=15)
plt.yscale('log')
plt.xlabel("Centrality", fontsize=20, loc='right')
plt.ylabel(r'$\phi = \frac{\sigma^2_{X}}{\sigma^2_{b}}$', fontsize=20)
plt.xticks(ticks=np.linspace(0, len(phi[0][2:]) - 1, len(phi[0][2:])), labels=x_cent, rotation=45,
           fontsize=15)
plt.yticks(fontsize=15)
# plt.show()
plt.close()

pro_bins = np.load('pro_bins.npy', allow_pickle=True)
pro_counts = np.load('pro_counts.npy', allow_pickle=True)[2]
b_counts = pro_counts[2]
b_cent = centrality[0]
byarr = pro_bins[2][2][0][:-1]
pro_counts = np.delete(pro_counts, 2, axis=0)
centrality_ = np.delete(centrality, 0, axis=0)
xarr = pro_bins[2][0][1][:-1]
yarr = pro_bins[2][0][0][:-1]
arr_pro = []
for i in range(len(pro_counts)):
    arr_pro.append([])
    index = np.where(yarr <= centrality_[i][0])[0][-1]
    arr = np.sum(pro_counts[i][:index], axis=0)
    arr[arr < 5] = 0  # Just to clean up noise in the plots.
    arr_pro[i].append(arr)
    for j in range(len(centrality_[0]) - 1):
        index = (np.where(yarr <= centrality_[i][j + 1])[0][-1],
                 np.where(yarr > centrality_[i][j])[0][0])
        arr = np.sum(pro_counts[i][index[1]:index[0]], axis=0)
        arr[arr < 5] = 0
        arr_pro[i].append(arr)
    index = np.where(yarr > centrality_[i][-1])[0][0]
    arr = np.sum(pro_counts[i][index:], axis=0)
    arr[arr < 5] = 0
    arr_pro[i].append(arr)
arr_pro.append([])

index = np.where(byarr > b_cent[0])[0][0]
arr = np.sum(b_counts[index:], axis=0)
arr[arr < 5] = 0
arr_pro[-1].append(arr)
for i in range(len(b_cent) - 1):
    index = (np.where(byarr > b_cent[i + 1])[0][0],
             np.where(byarr <= b_cent[i])[0][-1])
    arr = np.sum(b_counts[index[0]:index[1]], axis=0)
    arr[arr < 5] = 0
    arr_pro[-1].append(arr)
index = np.where(byarr <= b_cent[-1])[0][-1]
arr = np.sum(b_counts[:index], axis=0)
arr[arr < 5] = 0
arr_pro[-1].append(arr)

labels_short = labels[1:-1]
labels_short.append('b')
markers = ['o', '8', 's', 'P', '*', 'X', '$b$']
color = ['orangered', 'orange', 'black', 'blue', 'purple', 'darkviolet', 'brown']
fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
sets = [0, 1, 2, 5, 6]
for i in range(2):
    for j in range(2):
        x = i * 2 + j
        a = 0
        if x == 1:
            a = 3
        if x == 2:
            a = 6
        if x == 3:
            a = 8
        if x == 4:
            a = 10
        for k in sets:
            if k == 6:
                alpha = 1
            else:
                alpha = 0.5
            ax[i, j].plot(xarr, arr_pro[k][a], color=color[k], marker=markers[k], ms=10, label=labels_short[k],
                          alpha=alpha)
            ax[i, j].set_yscale('log')
            ax[i, j].legend()
            ax[i, j].set_title(x_cent[a] + " Centrality", fontsize=15)
            ax[i, j].set_xlabel(r"$\Delta N_p$", fontsize=15, loc='right')
            ax[i, j].set_ylabel(r"Count", fontsize=15, loc='top')
# plt.show()
plt.close()

# Trying something new here; it's not pretty (and I marred up the one above, too), but I need a plot!
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
textstr1 = '\n'.join(('60-70%', 'Centrality'))
textstr2 = '\n'.join(('0-5%', 'Centrality'))

fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
ax[0, 0].plot(xarr, arr_pro[0][1], color='orangered', marker='o', ms=10, alpha=0.5)
ax[0, 0].plot(xarr, arr_pro[-1][1], color='brown', marker='$b$', ms=10, alpha=0.5)
ax[0, 0].plot(xarr, arr_pro[0][-1], color='orangered', marker='o', ms=10, label=r'$X_{RM3}$', alpha=0.5)
ax[0, 0].plot(xarr, arr_pro[-1][-1], color='brown', marker='$b$', ms=10, label=r'b', alpha=0.5)
ax[0, 0].set_yscale('log')
ax[0, 0].legend()
ax[0, 0].set_xlabel(r"$\Delta N_p$", fontsize=15, loc='right')
ax[0, 0].set_ylabel(r"Count", fontsize=15, loc='top')
ax[0, 0].text(0.20, 0.95, textstr1, transform=ax[0, 0].transAxes, fontsize=20,
              verticalalignment='top', bbox=props)
ax[0, 0].text(0.65, 0.20, textstr2, transform=ax[0, 0].transAxes, fontsize=20,
              verticalalignment='top', bbox=props)

ax[0, 1].plot(xarr, arr_pro[1][1], color='orange', marker='o', ms=10, alpha=0.5)
ax[0, 1].plot(xarr, arr_pro[-1][1], color='brown', marker='$b$', ms=10, alpha=0.5)
ax[0, 1].plot(xarr, arr_pro[1][-1], color='orange', marker='o', ms=10, label=r'$X_{RM1}$', alpha=0.5)
ax[0, 1].plot(xarr, arr_pro[-1][-1], color='brown', marker='$b$', ms=10, label=r'b', alpha=0.5)
ax[0, 1].set_yscale('log')
ax[0, 1].legend()
ax[0, 1].set_xlabel(r"$\Delta N_p$", fontsize=15, loc='right')
ax[0, 1].set_ylabel(r"Count", fontsize=15, loc='top')
ax[0, 1].text(0.20, 0.95, textstr1, transform=ax[0, 1].transAxes, fontsize=20,
              verticalalignment='top', bbox=props)
ax[0, 1].text(0.65, 0.20, textstr2, transform=ax[0, 1].transAxes, fontsize=20,
              verticalalignment='top', bbox=props)

ax[1, 0].plot(xarr, arr_pro[2][1], color='black', marker='s', ms=10, alpha=0.5)
ax[1, 0].plot(xarr, arr_pro[-1][1], color='brown', marker='$b$', ms=10, alpha=0.5)
ax[1, 0].plot(xarr, arr_pro[2][-1], color='black', marker='s', ms=10, label=r'$X_{LW}$', alpha=0.5)
ax[1, 0].plot(xarr, arr_pro[-1][-1], color='brown', marker='$b$', ms=10, label=r'b', alpha=0.5)
ax[1, 0].set_yscale('log')
ax[1, 0].legend()
ax[1, 0].set_xlabel(r"$\Delta N_p$", fontsize=15, loc='right')
ax[1, 0].set_ylabel(r"Count", fontsize=15, loc='top')
ax[1, 0].text(0.20, 0.95, textstr1, transform=ax[1, 0].transAxes, fontsize=20,
              verticalalignment='top', bbox=props)
ax[1, 0].text(0.65, 0.20, textstr2, transform=ax[1, 0].transAxes, fontsize=20,
              verticalalignment='top', bbox=props)

ax[1, 1].plot(xarr, arr_pro[-2][1], color='darkviolet', marker='X', ms=10, alpha=0.5)
ax[1, 1].plot(xarr, arr_pro[-1][1], color='brown', marker='$b$', ms=10, alpha=0.5)
ax[1, 1].plot(xarr, arr_pro[-2][-1], color='darkviolet', marker='X', ms=10, label=r'$X_{CNN}$', alpha=0.5)
ax[1, 1].plot(xarr, arr_pro[-1][-1], color='brown', marker='$b$', ms=10, label=r'b', alpha=0.5)
ax[1, 1].set_yscale('log')
ax[1, 1].legend()
ax[1, 1].set_xlabel(r"$\Delta N_p$", fontsize=15, loc='right')
ax[1, 1].set_ylabel(r"Count", fontsize=15, loc='top')
ax[1, 1].text(0.20, 0.95, textstr1, transform=ax[1, 1].transAxes, fontsize=20,
              verticalalignment='top', bbox=props)
ax[1, 1].text(0.65, 0.20, textstr2, transform=ax[1, 1].transAxes, fontsize=20,
              verticalalignment='top', bbox=props)

plt.show()
plt.close()
