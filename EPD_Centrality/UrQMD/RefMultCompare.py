import uproot as up
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import colors as mcolors
from matplotlib import cm

# Centrality ranges, as follows:
# 90-100%, 80-90%, 70-80%, 60-70%, 50-60%,
# 40-50%, 30-40%, 20-30%, 10-20%, 5-10%, 0-5%
ranges = np.empty(10)
for i in range(8):
    ranges[i+2] = 0.1*(i+2)
for i in range(2):
    ranges[i] = 0.05*(i+1)

# Import the arrays.
os.chdir(r'D:\UrQMD_cent_sim\19')

urqmd = up.open('eta_b_full.root')  # For my data.
urqmd1 = up.open('OutputHist.root')  # For Rosi's data.

# First, Rosi's data; because it's easier becasue she's
# smarter than I am.
cRanges = ['0to5', '5to10', '10to20', '20to30', '30to40',
           '40to50', '50to60', '60to70', '70to80',
           '80to90', '90to100']
rosiRefs = [[], [], [], [], [], [], []]
for i in range(int(len(cRanges))):
    rosiRefs[0].append(np.asarray(
        urqmd1['hBParticleMult_%s' % cRanges[i]].numpy()))
    rosiRefs[1].append(np.asarray(
        urqmd1['hBRefMult1_%s' % cRanges[i]].numpy()))
    rosiRefs[2].append(np.asarray(
        urqmd1['hBRefMult2_%s' % cRanges[i]].numpy()))
    rosiRefs[3].append(np.asarray(urqmd1['hBFwd1_%s' % cRanges[i]].numpy()))
    rosiRefs[4].append(np.asarray(urqmd1['hBFwd2_%s' % cRanges[i]].numpy()))
    rosiRefs[5].append(np.asarray(urqmd1['hBFwd3_%s' % cRanges[i]].numpy()))
    rosiRefs[6].append(np.asarray(urqmd1['hBFwdAllp_%s' % cRanges[i]].numpy()))
rosiRefs = np.asarray(rosiRefs)

# Now for mine (which I made easier by using an array/ntuple).
skiRefs = np.loadtxt('arrayUrQMD.txt')

values = [[], [], [], [], [], [], []]
bins = [[], [], [], [], [], [], []]
centRanges = np.zeros((7, 10))
values[0], bins[0], _ = plt.hist(skiRefs[:, 0], bins=480, range=(0, 16))
plt.close()
for i in range(1, 7):
    values[i], bins[i], _ = plt.hist(
        skiRefs[:, i], bins=480, range=(0, 480))
    plt.close()
values, bins = np.asarray(values), np.asarray(bins)

for i in range(int(len(values[0]))):
    for j in range(10):
        if np.sum(values[0][: i]) >= np.sum(values[0])*ranges[j] and centRanges[0][j] == 0:
            centRanges[0][j] = bins[0][i]

for k in range(1, 7):
    for i in range(int(len(values[k]))):
        for j in range(10):
            if np.sum(values[k][-i-1:]) >= np.sum(values[k])*ranges[j] and centRanges[k][-j-1] == 0:
                centRanges[k][-j-1] = bins[k][-i-1]

# Distributions of b in each centrality range for each RefMult.
bDists = [[], [], [], [], [], [], []]
for i in range(7):
    for j in range(11):
        bDists[i].append([])

for i in range(int(len(skiRefs[:, 0]))):
    if skiRefs[i][0] <= centRanges[0][0]:
        bDists[0][0].append(skiRefs[i][0])
    for j in range(1, 10):
        if skiRefs[i][0] > centRanges[0][j-1] and skiRefs[i][0] <= centRanges[0][j]:
            bDists[0][j].append(skiRefs[i][0])
    if skiRefs[i][0] > centRanges[0][9]:
        bDists[0][10].append(skiRefs[i][0])

for i in range(1, 7):
    for j in range(int(len(skiRefs[:, 0]))):
        if skiRefs[j][i] <= centRanges[i][0]:
            bDists[i][10].append(skiRefs[j][0])
        for k in range(1, 10):
            if skiRefs[j][i] > centRanges[i][k-1] and skiRefs[j][i] <= centRanges[i][k]:
                bDists[i][-k-1].append(skiRefs[j][0])
        if skiRefs[j][i] > centRanges[i][9]:
            bDists[i][0].append(skiRefs[j][0])
bDists = np.asarray(bDists)

bVars = [[], [], [], [], [], [], []]
for i in range(11):
    for j in range(7):
        bVars[j].append(np.var(bDists[j][-i-1])/np.var(bDists[0][-i-1]))
print(centRanges)
print(bVars)

#+++++++++++++++++++++++++#
#:::::::::::::::::::::::::#
# Time to make some plots #
#:::::::::::::::::::::::::#
#+++++++++++++++++++++++++#

# Full distributions by centrality range for my data set.
Title1 = ['RefMult1', 'Refmult2', 'Fwd1', 'Fwd2', 'Fwd3', 'FwdAll']
# Plot the b distributions.
fig, axes = plt.subplots(nrows=2, ncols=3)

for i in range(2):
    for j in range(3):
        x = i*3 + j
        if x < 6:
            axes[i, j].hist2d(skiRefs[:, 0], skiRefs[:, x+1],
                              bins=[100, 100], norm=mcolors.LogNorm())
            axes[i, j].set_title(Title1[x], fontsize=20)
            axes[i, j].set_ylim(0, 500)
fig.suptitle("RefMult v b", fontsize=30)
plt.show()
plt.close()

colours = ['k', 'c', 'g', 'm', 'g', 'b']
linestyles = ['-', '-.', '-.', '-.', '-', '--']
legend = ['RefMult1', 'RefMult2', 'Fwd1', 'Fwd2', 'Fwd3', 'FwdAll']
for i in range(6):
    plt.hist(skiRefs[:, i+1], bins=480, range=(0, 480), histtype='step',
             linewidth=3.0, color=colours[i], linestyle=linestyles[i], label=legend[i])
plt.title("RefMults and b", fontsize=30)
plt.xlabel('RefMult', fontsize=20)
plt.ylabel('Counts', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=20)
plt.show()

Title = ['0-5%', '5-10%', '10-20%', '20-30%', '30-40%',
         '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
legend1 = ['b', 'RefMult1', 'RefMult2', 'Fwd1', 'Fwd2', 'Fwd3', 'FwdAll']
vals = [[], [], [], [], [], [], []]
bns = [[], [], [], [], [], [], []]
for j in range(7):
    for i in range(11):
        vals[j].append([])
        bns[j].append([])
fig, axes = plt.subplots(nrows=4, ncols=3)
for i in range(4):
    for j in range(3):
        x = i*3 + j
        if x < 11:
            for k in range(7):
                vals[k][x], bns[k][x], _ = plt.hist(bDists[k][x], bins=100)
plt.close()

vals = np.asarray(vals)
bns = np.asarray(bns)
fig, axes = plt.subplots(nrows=4, ncols=3)
for i in range(4):
    for j in range(3):
        x = i*3 + j
        if x < 11:
            axes[i, j].semilogy(bns[1][x][1:], vals[1][x] /
                                np.sum(vals[1][x]), label='Skipper')
            axes[i, j].semilogy(rosiRefs[1][x][1][:-1],
                                rosiRefs[1][x][0], label='Rosi')
            axes[i, j].set_title(Title[x])
fig.suptitle("RefMult1", fontsize=30)
plt.show()
plt.close()

fig, axes = plt.subplots(nrows=4, ncols=3)
for i in range(4):
    for j in range(3):
        x = i*3 + j
        if x < 11:
            axes[i, j].semilogy(bns[2][x][1:], vals[2][x] /
                                np.sum(vals[2][x]), label='Skipper')
            axes[i, j].semilogy(rosiRefs[2][x][1][:-1],
                                rosiRefs[2][x][0], label='Rosi')
            axes[i, j].set_title(Title[x])
fig.suptitle("RefMult2", fontsize=30)
plt.show()
plt.close()

fig, axes = plt.subplots(nrows=4, ncols=3)
for i in range(4):
    for j in range(3):
        x = i*3 + j
        if x < 11:
            axes[i, j].semilogy(bns[3][x][1:], vals[3][x] /
                                np.sum(vals[3][x]), label='Skipper')
            axes[i, j].semilogy(rosiRefs[3][x][1][:-1],
                                rosiRefs[3][x][0], label='Rosi')
            axes[i, j].set_title(Title[x])
fig.suptitle("Fwd1", fontsize=30)
plt.show()
plt.close()

fig, axes = plt.subplots(nrows=4, ncols=3)
for i in range(4):
    for j in range(3):
        x = i*3 + j
        if x < 11:
            axes[i, j].semilogy(bns[4][x][1:], vals[4][x] /
                                np.sum(vals[4][x]), label='Skipper')
            axes[i, j].semilogy(rosiRefs[4][x][1][:-1],
                                rosiRefs[4][x][0], label='Rosi')
            axes[i, j].set_title(Title[x])
fig.suptitle("Fwd2", fontsize=30)
plt.show()
plt.close()

fig, axes = plt.subplots(nrows=4, ncols=3)
for i in range(4):
    for j in range(3):
        x = i*3 + j
        if x < 11:
            axes[i, j].semilogy(bns[5][x][1:], vals[5][x] /
                                np.sum(vals[5][x]), label='Skipper')
            axes[i, j].semilogy(rosiRefs[5][x][1][:-1],
                                rosiRefs[5][x][0], label='Rosi')
            axes[i, j].set_title(Title[x])
fig.suptitle("Fwd3", fontsize=30)
plt.show()
plt.close()

fig, axes = plt.subplots(nrows=4, ncols=3)
for i in range(4):
    for j in range(3):
        x = i*3 + j
        if x < 11:
            axes[i, j].semilogy(bns[6][x][1:], vals[6][x] /
                                np.sum(vals[6][x]), label='Skipper')
            axes[i, j].semilogy(rosiRefs[6][x][1][:-1],
                                rosiRefs[6][x][0], label='Rosi')
            axes[i, j].set_title(Title[x])
fig.suptitle("FwdAll", fontsize=30)
plt.show()
plt.close()

fig, ax = plt.subplots()
xLabels = ['90-100%', '80-90%', '70-80%', '60-70%', '50-60%',
           '40-50%', '30-40%', '20-30%', '10-20%', '5-10%', '0-5%']
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
s = 20
marker = ['.', 's', '<', '^', 'p', 'X']
mec = ['blue', 'black', 'purple', 'red', 'blue', 'black']
for i in range(1, 7):
    plt.plot(x, bVars[i], linewidth=3, color=mec[i-1], marker=marker[i-1],
             ms=s, mfc='none', mec=mec[i-1])
plt.xticks(x, xLabels, rotation=60)
leg = plt.legend(legend, fontsize=15)
plt.title(r'$\Phi$ for 19.6 GeV', fontsize=30)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Centrality class (%)', fontsize=20)
plt.yscale('log')
plt.ylabel(
    r'$\Phi$ = $\sigma^{2}_{b_{X}}$/$\sigma^{2}_{b_{centrality-b}}$', fontsize=20)
plt.show()
