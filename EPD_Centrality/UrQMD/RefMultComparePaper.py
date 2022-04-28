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

# Import the arrays from the paper.
os.chdir(r'D:\UrQMD_cent_sim\PaperFiles\En_19\e_by_e')
files = up.open('ppkset_rap_19.root')  # For my data.
urqmd = files['AMPT_tree']

refMult = []
for i in range(7):
    refMult.append([])

refMult[0] = urqmd.array('Imp')
refMult[1] = urqmd.array('RefMult')
refMult[2] = urqmd.array('RefMult2')
#refMult[3] = urqmd.array('RefMult3')
refMult[3] = urqmd.array('RefMultEPD1')
refMult[4] = urqmd.array('RefMultEPD2')
refMult[5] = urqmd.array('RefMultEPD3')
refMult[6] = urqmd.array('RefMultEPDfull')

# Now to get centrality ranges.
r = int(len(refMult))
centRanges = np.zeros((r, 10))
values = []
bins = []
for i in range(r):
    values.append([])
    bins.append([])
values[0], bins[0], _ = plt.hist(refMult[0], bins=480, range=(0, 16))
plt.close()
for i in range(1, r):
    values[i], bins[i], _ = plt.hist(
        refMult[i], bins=480, range=(0, 480))
    plt.close()
values, bins = np.asarray(values), np.asarray(bins)
for i in range(int(len(values[0]))):
    for j in range(10):
        if np.sum(values[0][: i]) >= np.sum(values[0])*ranges[j] and centRanges[0][j] == 0:
            centRanges[0][j] = bins[0][i]

for k in range(1, r):
    for i in range(int(len(values[k]))):
        for j in range(10):
            if np.sum(values[k][-i-1:]) >= np.sum(values[k])*ranges[j] and centRanges[k][-j-1] == 0:
                centRanges[k][-j-1] = bins[k][-i-1]

# Distributions of b in each centrality range for each RefMult.
bDists = []
for i in range(r):
    bDists.append([])
    for j in range(11):
        bDists[i].append([])

for i in range(int(len(refMult[0]))):
    if refMult[0][i] <= centRanges[0][0]:
        bDists[0][0].append(refMult[0][i])
    for j in range(1, 10):
        if refMult[0][i] > centRanges[0][j-1] and refMult[0][i] <= centRanges[0][j]:
            bDists[0][j].append(refMult[0][i])
    if refMult[0][i] > centRanges[0][9]:
        bDists[0][10].append(refMult[0][i])

for i in range(1, r):
    for j in range(int(len(refMult[0]))):
        if refMult[i][j] <= centRanges[i][0]:
            bDists[i][10].append(refMult[0][j])
        for k in range(1, 10):
            if refMult[i][j] > centRanges[i][k-1] and refMult[i][j] <= centRanges[i][k]:
                bDists[i][-k-1].append(refMult[0][j])
        if refMult[i][j] > centRanges[i][9]:
            bDists[i][0].append(refMult[0][j])
bDists = np.asarray(bDists)

bVars = []
for i in range(r):
    bVars.append([])
for i in range(11):
    for j in range(r):
        bVars[j].append(np.var(bDists[j][-i-1])/np.var(bDists[0][-i-1]))
print(centRanges)
# print(bVars)

#+++++++++++++++++++++++++#
#:::::::::::::::::::::::::#
# Time to make some plots #
#:::::::::::::::::::::::::#
#+++++++++++++++++++++++++#

# Full distributions by centrality range for my data set.
legend = ['RefMult1', 'Refmult2', 'Fwd1',
          'Fwd2', 'Fwd3', 'FwdAll']
# Plot the b distributions.
fig, axes = plt.subplots(nrows=2, ncols=3)

for i in range(2):
    for j in range(3):
        x = i*3 + j
        if x < r-1:
            axes[i, j].hist2d(refMult[0], refMult[x+1],
                              bins=[100, 100], norm=mcolors.LogNorm())
            axes[i, j].set_title(legend[x], fontsize=20)
            axes[i, j].set_ylim(0, 500)
fig.suptitle("RefMult v b", fontsize=30)
plt.show()
plt.close()

colours = ['k', 'c', 'g', 'm', 'g', 'b', 'deeppink', 'yellow']
linestyles = ['-', '-.', '-.', '-.', '-', '--', '-', '-']
for i in range(r-1):
    plt.hist(refMult[i+1], bins=480, range=(0, 480), histtype='step',
             linewidth=3.0, color=colours[i], linestyle=linestyles[i], label=legend[i])
plt.title("RefMults and b", fontsize=30)
plt.xlabel('RefMult', fontsize=20)
plt.ylabel('Counts', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=20)
plt.show()
plt.close()

fig, ax = plt.subplots()
xLabels = ['90-100%', '80-90%', '70-80%', '60-70%', '50-60%',
           '40-50%', '30-40%', '20-30%', '10-20%', '5-10%', '0-5%']
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
s = 20
marker = ['.', 's', '<', '^', 'p', 'X', 'x', 'd']
mec = ['blue', 'black', 'purple', 'red', 'blue', 'black', 'green', 'orange']
for i in range(1, r):
    plt.plot(x, bVars[i], linewidth=3, color=mec[i-1], marker=marker[i-1],
             ms=s, mfc='none', mec=mec[i-1])
plt.xticks(x, xLabels, rotation=60)
leg = plt.legend(legend, fontsize=15)
plt.title(r'$\Phi$ for 19.6 GeV', fontsize=30)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Centrality class (%)', fontsize=20)
plt.yscale('log')
plt.ylim((0, 50))
plt.ylabel(
    r'$\Phi$ = $\sigma^{2}_{b_{X}}$/$\sigma^{2}_{b_{centrality-b}}$', fontsize=20)
plt.show()
