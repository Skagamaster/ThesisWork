import numpy as np
import matplotlib.pyplot as plt
import os

energy = str(200)
col = energy
if energy == '19':
    col = '19.6'
if energy == '7':
    col = '7.7'

inFile1 = 'bVars%s.txt' % energy
inFile2 = 'bVarsRosi%s.txt' % energy

os.chdir(r'D:\UrQMD_cent_sim')
data1 = np.loadtxt(inFile1)
data2 = np.loadtxt(inFile2)
names = ["RefMult1", "RefMult1'", "RefMult2", "RefMult2'", "RefMult3",
         "RefMult3'", "Fwd1", "Fwd1'", "Fwd2", "Fwd2'", "Fwd3", "Fwd3'",
         "FwdAll", "FwdAll'"]
s = [0, 1, 2, 3, 4, 5, 6]
fig, ax = plt.subplots()
xLabels = ['90-100%', '80-90%', '70-80%', '60-70%', '50-60%',
           '40-50%', '30-40%', '20-30%', '10-20%', '5-10%', '0-5%']
n = 20
marker = ['o', 'v', '^', '<', '>', 's', '8', 'D']
mec = ['blue', 'black', 'purple', 'red',
       'blueviolet', 'gray', 'green', 'orange']
x = np.linspace(0, 1, 11)
y1 = np.linspace(2, 10, 5)
y2 = np.linspace(20, 100, 5)
y3 = np.linspace(200, 1000, 4)
y = np.hstack((y1, y2, y3))
ymax = 0
for i in range(int(len(data1))):
    for i in s:
        plt.plot(x, data1[i], marker=marker[i],
                 ms=20, mfc='none', mec=mec[i], c='none')
        plt.plot(x, data2[i], marker=marker[i],
                 ms=10, mec=mec[i], mfc=mec[i], c='none')
        plt.yscale('log')
        if np.max(data1[i]) > ymax:
            ymax = np.max(data1[i])
        if np.max(data1[i]) > ymax:
            ymax = np.max(data1[i])
plt.hlines(1.0, -0.1, 1.1, linestyle=':', color='black')
leg = plt.legend(names)
plt.title(r'$\Phi$ Difference at $\sqrt{s_{NN}}$= %s GeV' % col, fontsize=30)
plt.xticks(fontsize=15)
plt.yticks(ticks=y, labels=y, fontsize=15)
plt.xlim(-0.1, 1.1)
plt.ylim(0.9, 2*ymax)
plt.xlabel('Centrality class (%)', fontsize=20)
plt.ylabel(
    r'$\Phi$ = $\sigma^{2}_{b_{X}}$/$\sigma^{2}_{b_{centrality-b}}$', fontsize=20)
plt.xticks(x, xLabels, rotation=60)
plt.show()
