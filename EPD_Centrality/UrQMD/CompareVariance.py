import numpy as np
import matplotlib.pyplot as plt
import os

file = np.loadtxt(r'D:\UrQMD_cent_sim\19\Var_Deltas_rms.txt')
varComp = [[], [], [], [], []]
for i in range(5):
    for j in range(9):
        varComp[i].append(file[j+9*i])
varComp = np.asarray(varComp)

A = varComp[:, :, 0]
B = varComp[:, :, 1]
C = A/B*100-100
print(C[0])

fig = plt.figure()
ax = fig.add_subplot(111)
markers = ['*', 's', 'p', 'h', '8', 'P', 'd', 'O', 'X']
colors = ['b', 'g', 'r', 'c', 'k']
Title1 = ['0-5%', '5-10%', '10-20%', '20-30%',
          '30-40%', '40-50%', '50-60%', '60-70%', '70-80%']
x = range(9)
xLabels = []
for i in range(int(len(Title1))):
    xLabels.append(Title1[-i-1])
legend1 = ['refmult1', 'refmult2', 'Fwd1', 'Fwd2', 'Fwd3']
for j in range(5):
    plt.plot(x, C[j], c=colors[j], marker=markers[j],
             ms=30, mfc='none', mew=3, lw=3)
leg = plt.legend(legend1, fontsize=20)
plt.xticks(x, xLabels, rotation=60, fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Centrality range', fontsize=20)
plt.ylabel(r'$\Delta$ for $\sigma^{2}$ values (%)', fontsize=20)
plt.title(
    r'Differences in $\sigma^{2}$: Rosi v Skipper (rms method)', fontsize=30)
plt.show()
