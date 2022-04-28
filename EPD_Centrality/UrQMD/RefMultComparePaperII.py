import uproot as up
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import colors as mcolors
from matplotlib import cm

# Centrality ranges, as follows:
# 90-100%, 80-90%, 70-80%, 60-70%, 50-60%,
# 40-50%, 30-40%, 20-30%, 10-20%, 5-10%, 0-5%
ranges = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ranges1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
ranges = np.asarray(ranges)
ranges1 = np.asarray(ranges1)

# Import the arrays from the paper.
os.chdir(r'D:\UrQMD_cent_sim\PaperFiles\En_19\e_by_e')
files = up.open('ppkset_rap_19.root')  # For my data.
urqmd = files['AMPT_tree']

refMult = []
for i in range(8):
    refMult.append([])

refMult[0] = urqmd.array('Imp')
refMult[1] = urqmd.array('RefMult')
refMult[2] = urqmd.array('RefMult2')
refMult[3] = urqmd.array('RefMult3')
refMult[4] = urqmd.array('RefMultEPD1')
refMult[5] = urqmd.array('RefMultEPD2')
refMult[6] = urqmd.array('RefMultEPD3')
refMult[7] = urqmd.array('RefMultEPDfull')

# Now to get centrality ranges.
r = int(len(refMult))
centRanges = np.zeros((r, 10))

for i in range(int(len(ranges))):
    centRanges[0][i] = np.quantile(refMult[0], ranges[i])

for i in range(int(len(ranges1))):
    for j in range(1, r):
        centRanges[j][i] = np.quantile(refMult[j], ranges1[i])

print(centRanges)

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
bVars = np.asarray(bVars)
print(bVars)

np.savetxt('bVars19.txt', bVars)

#+++++++++++++++++++++++++#
#:::::::::::::::::::::::::#
# Time to make some plots #
#:::::::::::::::::::::::::#
#+++++++++++++++++++++++++#

# Full distributions by centrality range for my data set.
legend = ['RefMult1', 'Refmult2', 'Fwd1',
          'Fwd2', 'Fwd3', 'FwdAll']
# Plot the b distributions vs refMults.
fig, axes = plt.subplots(nrows=2, ncols=3)

for i in range(2):
    for j in range(3):
        x = i*3 + j
        if x < r-1:
            axes[i, j].hist2d(refMult[0], refMult[x+1],
                              bins=[100, 100], norm=mcolors.LogNorm())
            axes[i, j].set_title(legend[x], fontsize=20)
            axes[i, j].set_ylim(0, 500)
fig.suptitle("RefMult v b, 7.7 GeV", fontsize=30)
plt.show()
plt.close()

# Plot the centrality.
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

# Plot the sigma ratios.
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
plt.title(r'$\Phi$ for 7.7 GeV', fontsize=30)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Centrality class (%)', fontsize=20)
plt.yscale('log')
plt.ylim((0, int(2*np.max(bVars))))
plt.ylabel(
    r'$\Phi$ = $\sigma^{2}_{b_{X}}$/$\sigma^{2}_{b_{centrality-b}}$', fontsize=20)
plt.show()
plt.close()

# Plot the b distributions for the various refMults.
fig, axes = plt.subplots(nrows=3, ncols=4)
legend = ['b', 'RefMult1', 'Refmult2', 'Fwd1',
          'Fwd2', 'Fwd3', 'FwdAll']
titles = ['0-5%', '5-10%', '10-20%', '20-30%', '30-40%',
          '40-50%', '50-60%', '60-70%', '70-80%,', '80-90%', '90-100%']
for i in range(3):
    for j in range(4):
        x = int(i*4+j)
        if x < 11:
            for k in range(r):
                axes[i, j].hist(bDists[k][x], bins=50, label=legend[k],
                                density=True, histtype='step')
                axes[i, j].set_title(titles[x], fontsize=20)
for i in range(r):
    axes[2, 3].plot(0)
axes[2, 3].legend(legend)
plt.show()
plt.close()
