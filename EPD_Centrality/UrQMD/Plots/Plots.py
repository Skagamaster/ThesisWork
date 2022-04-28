import os
import uproot as up
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

# Import the arrays from the paper.
os.chdir(r'D:\UrQMD_cent_sim\PaperFiles\En_7\e_by_e')
files = up.open('ppkset_rap_7.root')  # For my data.
urqmd = files['AMPT_tree']
b = urqmd.array('Imp')

os.chdir(r'D:\UrQMD_cent_sim\7')
refMult = np.load('refMult.npy', allow_pickle=True)
bVars = np.load('bVars.npy', allow_pickle=True)
bDists = np.load('bDists.npy', allow_pickle=True)
centRanges = np.load('centRangesComp.npy', allow_pickle=True)

# THIS IS FOR THE PRESENTATION
##################################

plt.hist(b, bins=200, histtype='step', fill=False, lw=3)
plt.title('b Distribution', fontsize=30)
plt.xlabel('b (fm)', fontsize=20)
plt.ylabel('Counts', fontsize=20)
plt.show()
plt.close()

bPap = centRanges[0]
bUs = np.hstack((0, centRanges[8, :9]))

fig, axes = plt.subplots(nrows=3, ncols=3)
titles = ['0-5%', '5-10%', '10-20%', '20-30%', '30-40%',
          '40-50%', '50-60%', '60-70%', '70-80%']
for i in range(3):
    for j in range(3):
        x = int(i*3+j)
        if x < 8:
            axes[i, j].hist(b, range=[bPap[x], bPap[x+1]], bins=20, label='From Paper',
                            histtype='step', Fill=False, linewidth=3)
            axes[i, j].hist(b, range=[bUs[x], bUs[x+1]], bins=20, label='From Calculation',
                            histtype='step', Fill=False, linewidth=3)
            axes[i, j].set_title(titles[x], fontsize=20)
            axes[i, j].set_yscale('log')
            #axes[i, j].set_ylabel(r'$\frac{dN}{db}$', fontsize=15)
            #axes[i, j].set_xlabel('b (fm)', fontsize=15)
axes[2, 2].plot(0)
axes[2, 2].plot(0)
axes[2, 2].set_axis_off()
leg = axes[2, 2].legend(['From Paper', 'From Calculation'], fontsize=20)
for line in leg.get_lines():
    line.set_linewidth(3)
plt.show()
plt.close()

print(centRanges[9:16, 1:]-centRanges[1:8, :-1]+1)

##################################

r = int(len(refMult))
print(r)
legendMaster = ['b', 'RefMult1', 'RefMult2', 'RefMult3',
                'Fwd1', 'Fwd2', 'Fwd3', 'FwdAll', r'$FwdAll_{LW}$']

# s is the set of refmults you want to observe, order in legendMaster.
s = [0, 1, 2, 7, 8]
legend = []
for i in s:
    legend.append(legendMaster[i])

fig, axes = plt.subplots(2, 2)
for j in range(2):
    for i in range(2):
        x = i*2+j
        axes[j, i].hist2d(refMult[0], refMult[s[x+1]], bins=[100, 100], cmin=0.01,
                          norm=mcolors.LogNorm())
        axes[j, i].set_title(legendMaster[s[x+1]], fontsize=20)
        #axes[j,i].set_ylim(0, 500)
        axes[j, i].set_xlabel('b (fm)', fontsize=20)
        axes[j, i].set_ylabel(r'$N_{ch}$', fontsize=20)
plt.show()
plt.close()

# Plot the b distributions for the various refMults.
fig, axes = plt.subplots(nrows=3, ncols=4)
titles = ['0-5%', '5-10%', '10-20%', '20-30%', '30-40%',
          '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
for i in range(3):
    for j in range(4):
        x = int(i*4+j)
        if x < 11:
            for k in range(r):
                if k in s:
                    axes[i, j].hist(bDists[k][x], bins=50, label=legendMaster[k],
                                    density=True, histtype='step', linewidth=3)
                    axes[i, j].set_title(titles[x], fontsize=20)
                    axes[i, j].set_yscale('log')
                    #axes[i, j].set_ylabel(r'$\frac{dN}{db}$', fontsize=15)
                    #axes[i, j].set_xlabel('b (fm)', fontsize=15)

for i in range(r):
    if i in s:
        axes[2, 3].plot(0)
leg = axes[2, 3].legend(legend)
for line in leg.get_lines():
    line.set_linewidth(3)
plt.show()
plt.close()

# Plot the sigma ratios.
fig, ax = plt.subplots()
xLabels = ['90-100%', '80-90%', '70-80%', '60-70%', '50-60%',
           '40-50%', '30-40%', '20-30%', '10-20%', '5-10%', '0-5%']
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
n = 20
marker = ['.', 's', '<', '^', 'p', 'X', 'x', 'd']
mec = ['blue', 'black', 'purple', 'red',
       'blueviolet', 'gray', 'green', 'orange']
for i in range(1, r):
    if i in s:
        plt.plot(x, bVars[i], linewidth=3, color=mec[i-1], marker=marker[i-1],
                 ms=n, mfc='none', mec=mec[i-1])
plt.xticks(x, xLabels, rotation=60)
leg = plt.legend(legend[1:], fontsize=15)
plt.title(r'$\Phi$ for 200 GeV', fontsize=30)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Centrality class (%)', fontsize=20)
plt.yscale('log')
plt.ylim((0, int(2*np.max(bVars))))
plt.ylabel(
    r'$\Phi$ = $\sigma^{2}_{b_{X}}$/$\sigma^{2}_{b_{centrality-b}}$', fontsize=20)
plt.show()
plt.close()
