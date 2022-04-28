import os
import uproot as up
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

energy = str(19)
col = energy

# Here are the arrays for 7.7 GeV from the paper's plot:
ref3 = np.asarray([1.3, 2.2, 1.6, 2.2, 2.5, 2.0, 2.2, 3.3, 5.0])
fwdA = np.asarray([5.3, 20.2, 11.6, 21.9, 27.2, 23.8, 26.4, 35.1, 50.8])

if energy == '19':
    col = '19.6'
if energy == '7':
    col = '7.7'

os.chdir(r'D:\UrQMD_cent_sim\%s' % energy)
refMult = np.load('refMult.npy', allow_pickle=True)
bVarsRACF = np.load('bVarsBill1.npy', allow_pickle=True)
bVarsBill = np.load('bVarsBill.npy', allow_pickle=True)
bVarsCCNU = np.load('bVars1.npy', allow_pickle=True)
bVarsNew = np.load('bVarsNew.npy', allow_pickle=True)
bDists1 = np.load('bDists1.npy', allow_pickle=True)
bDists = np.load('bDistsBill1.npy', allow_pickle=True)
bDistsP = np.load('bDistsBill.npy', allow_pickle=True)
centRanges = np.load('centRangesBill1.npy', allow_pickle=True)
centRangesP = np.load('centRangesBill.npy', allow_pickle=True)
bBill = np.load('bBill.npy', allow_pickle=True)
bRACF = np.load('bRACF.npy', allow_pickle=True)
os.chdir(r'D:\UrQMD_cent_sim\PaperFiles\En_%s\e_by_e' % energy)
files = up.open('ppkset_rap_%s.root' % energy)  # For my data.
urqmd = files['AMPT_tree']
b = urqmd.array('Imp')

# THIS IS FOR THE PRESENTATION
##################################

plt.hist(bBill, bins=100, histtype='step',
         fill=False, lw=3, color='black', normed=True)
plt.hist(b, bins=100, histtype='step', fill=False,
         lw=3, color='purple', normed=True)
# plt.hist(bRACF, bins=100, histtype='step',
#         fill=False, lw=3, color='r', normed=True)
plt.title('b Distribution for %s GeV' % col, fontsize=30)
lgnd = ['b-Bill', 'b-CCNU']
plt.legend(lgnd, loc='upper left', fontsize=20)
plt.xlabel('b (fm)', fontsize=20)
plt.ylabel(r'$\frac{dN}{db}$', fontsize=20)
# plt.show()
plt.close()

bUs = centRanges[0]
bPap = centRangesP[0]
print(bUs, bPap)

fig, axes = plt.subplots(nrows=3, ncols=3)
titles = ['0-5%', '5-10%', '10-20%', '20-30%', '30-40%',
          '40-50%', '50-60%', '60-70%', '70-80%']
axes[0, 0].hist(b, range=[0, bPap[0]], bins=20,
                histtype='step', Fill=False, linewidth=3, color='b')
axes[0, 0].hist(b, range=[0, bUs[0]], bins=20,
                histtype='step', Fill=False, linewidth=3, color='r')
axes[0, 0].set_title(titles[0], fontsize=20)
axes[0, 0].set_yscale('log')
for i in range(3):
    for j in range(3):
        x = int(i*3+j)
        if x in range(1, 8):
            axes[i, j].hist(b, range=[bPap[x-1], bPap[x]], bins=20,
                            histtype='step', Fill=False, linewidth=3, color='b')
            axes[i, j].hist(b, range=[bUs[x-1], bUs[x]], bins=20,
                            histtype='step', Fill=False, linewidth=3, color='r')
            axes[i, j].set_title(titles[x], fontsize=20)
            axes[i, j].set_yscale('log')
axes[2, 2].plot(0, c='b')
axes[2, 2].plot(0, c='r')
axes[2, 2].set_axis_off()
leg = axes[2, 2].legend(['UrQMD-Bill', 'UrQMD-RACF'], fontsize=20)
for line in leg.get_lines():
    line.set_linewidth(3)
# plt.show()
plt.close()

r = int(len(bVarsBill))
legendMaster = ['b CCNU', 'RefMult1 CCNU', 'RefMult2 CCNU', 'RefMult3 CCNU',
                'Fwd1 CCNU', 'Fwd2 CCNU', 'Fwd3 CCNU', 'FwdAll CCNU']
legendMaster1 = ['b', 'RefMult1', 'RefMult2', 'RefMult3',
                 'Fwd1', 'Fwd2', 'Fwd3', 'FwdAll']

# s is the set of refmults you want to observe, order in legendMaster.
s = [7]
legend = []
for i in s:
    legend.append(legendMaster[i])
    legend.append(legendMaster1[i])

# Plot the b distributions for the various refMults.
fig, axes = plt.subplots(nrows=3, ncols=3)
titles = ['0-5%', '5-10%', '10-20%', '20-30%', '30-40%',
          '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
color = ['r', 'b']
for i in range(3):
    for j in range(3):
        x = int(i*3+j)
        if x < 8:
            # for k in range(r):
            #    if k in s:
            axes[i, j].hist(bDistsP[2][x], bins=50, label=legendMaster[1],
                            density=True, histtype='step', linewidth=2,
                            color='b')
            axes[i, j].hist(bDists[2][x], bins=50, label=legendMaster[1],
                            density=True, histtype='step', linewidth=4,
                            linestyle=':', color='r')
            axes[i, j].set_title(titles[x], fontsize=20)
            axes[i, j].set_yscale('log')
axes[2, 2].plot(0, c='b')
axes[2, 2].plot(0, c='r', linestyle='dashed')
axes[2, 2].set_axis_off()
leg = axes[2, 2].legend(['RefMult1-Bill', 'RefMult1-RACF'], fontsize=30)
for line in leg.get_lines():
    line.set_linewidth(4)
# plt.show()
plt.close()

fig, axes = plt.subplots(nrows=3, ncols=3)
titles = ['0-5%', '5-10%', '10-20%', '20-30%', '30-40%',
          '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
color = ['black', 'gold', 'green', 'red', 'green', 'pink', 'cyan', 'blue']
style = ['-', '-', ':', '--', '-.', '-.', '-.', '-.']
x = 0
for i in range(3):
    for j in range(3):
        #x = int(i*3+j)
        #x = 0
        list = [0, 1, 2, 3, 7, 8, 9, 10]
        if x < 8:
            for k in [4, 6, 7]:
                axes[i, j].hist(bDists[k][list[x]], bins=50, label=legendMaster[1],
                                density=True, histtype='step', linewidth=2,
                                color=color[k], linestyle=style[k])
                axes[i, j].hist(bDists[0][list[x]], bins=50, density=True,
                                histtype='step', linewidth=2, color='black')
                axes[i, j].set_title(titles[list[x]], fontsize=20)
                axes[i, j].set_yscale('log')
            x += 1
for i in range(1, r):
    axes[2, 2].plot(0, color=color[k], linestyle=style[k],
                    label=legendMaster1[k])
axes[2, 2].plot(0, color='grey', label='b')
legendMaster1.append('b')
leg = axes[2, 2].legend(legendMaster1[1:], fontsize=15, ncol=2)
for i in range(r-1):
    leg.get_lines()[i].set_linewidth(4)
    leg.get_lines()[i].set_color(color[i+1])
    leg.get_lines()[i].set_linestyle(style[i+1])
leg.get_lines()[r-1].set_linewidth(4)
leg.get_lines()[r-1].set_color('grey')
axes[2, 2].set_axis_off()
plt.show()
plt.close()

# Plot the sigma ratios.
legend = []
for i in s:
    legend.append(legendMaster[i])
    legend.append(legendMaster1[i])
# fig, ax = plt.subplots()
xLabels = ['90-100%', '80-90%', '70-80%', '60-70%', '50-60%',
           '40-50%', '30-40%', '20-30%', '10-20%', '5-10%', '0-5%']
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
n = 20
marker = ['.', 's', '<', '^', 'p', 'X', 'x', 'd']
mec = ['blue', 'black', 'purple', 'red',
       'blueviolet', 'gray', 'green', 'orange', 'gold']
plt.figure(figsize=(9, 9))
for i in [1, 7]:
    plt.plot(x[:], bVarsNew[i][:][::-1], linewidth=2, color=mec[i], marker=marker[0],
             ms=n, mfc='none', mec=mec[i])
    plt.plot(x[:], bVarsCCNU[i][:][::-1], linewidth=4, color=mec[i], marker=marker[1],
             ms=n, mfc='none', mec=mec[i], linestyle=':')
    # plt.plot(x[:], bVarsRACF[i][:][::-1], linewidth=3, color=mec[i], marker=marker[2],
    #         ms=n, mfc='none', mec=mec[i], linestyle='-.')
plt.xticks(x, xLabels, rotation=60)
legend = ['RefMult1-Llope', 'RefMult1-Arghya',
          'FwdAll-Llope', 'FwdAll-Arghya',
          'Fwd3-Llope', 'Fwd3-Arghya']
leg = plt.legend(legend, fontsize=15)
plt.title(r'$\Phi$ for %s GeV' % col, fontsize=30)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Centrality class (%)', fontsize=20)
plt.yscale('log')
plt.ylim((1, 200))
plt.ylabel(
    r'$\Phi$ = $\sigma^{2}_{b_{X}}$/$\sigma^{2}_{b_{centrality-b}}$', fontsize=20)
plt.tight_layout()
# plt.show()
plt.close()
