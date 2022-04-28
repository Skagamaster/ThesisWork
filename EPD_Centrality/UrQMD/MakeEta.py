import uproot as up
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(r'D:\UrQMD_cent_sim\19')

urqmd = up.open('eta_b.root')

# Make the eta and b arrays (1D).
refmult1 = np.asarray(urqmd["refmult1"].numpy())
refmult2 = np.asarray(urqmd["refmult2"].numpy())
Fwd1 = np.asarray(urqmd['Fwd1'].numpy())
Fwd2 = np.asarray(urqmd['Fwd2'].numpy())
Fwd3 = np.asarray(urqmd['Fwd3'].numpy())
refmults = []
'''
FwdAll = []
FwdAll.append([])
FwdAll.append([])
FwdAll[0] = Fwd1[0][:]+Fwd2[0][:]+Fwd3[0][:]
FwdAll[1] = Fwd1[1]
FwdAll = np.asarray(FwdAll)
'''
b = np.asarray(urqmd[b"setB"].numpy())
refmults.append(b[0])
refmults.append(refmult1[0])
refmults.append(refmult2[0])
refmults.append(Fwd1[0])
refmults.append(Fwd2[0])
refmults.append(Fwd3[0])
refmults = np.asarray(refmults)

# Make the eta vs b arrays (2D).
refVb = [[], [], [], [], []]
bVr1 = (np.asarray(urqmd['EtaVb0'].numpy()))
bVr2 = np.asarray(urqmd['EtaVb1'].numpy())
bVf1 = np.asarray(urqmd['EtaVb2'].numpy())
bVf2 = np.asarray(urqmd['EtaVb3'].numpy())
bVf3 = np.asarray(urqmd['EtaVb4'].numpy())

refVb[0].append(bVr1[1][0][1][:-1])
for i in range(int(len(bVr1[0]))):
    refVb[0].append(bVr1[0][i])
refVb[1].append(bVr2[1][0][1][:-1])
for i in range(int(len(bVr2[0]))):
    refVb[1].append(bVr2[0][i])
refVb[2].append(bVf1[1][0][1][:-1])
for i in range(int(len(bVf1[0]))):
    refVb[2].append(bVf1[0][i])
refVb[3].append(bVf2[1][0][1][:-1])
for i in range(int(len(bVf2[0]))):
    refVb[3].append(bVf2[0][i])
refVb[4].append(bVf3[1][0][1][:-1])
for i in range(int(len(bVf3[0]))):
    refVb[4].append(bVf3[0][i])

refVb = np.asarray(refVb)

# Centrality bins.  ***MAKE TOTALS FOR EACH BIN!!***
total = []
for i in range(6):
    total.append(np.sum(refmults[i]))
total = np.asarray(total)

# Centrality ranges, as follows:
# 70-80%, 60-70%, 50-60%, 40-50%,
# 30-40%, 20-30%, 10-20%, 5-10%, 0-5%
ranges = np.empty(9)
for i in range(7):
    ranges[i+2] = 0.1*(i+2)
for i in range(2):
    ranges[i] = 0.05*(i+1)

# Points for centrality range points for each refmult.
cents = np.zeros((6, 9))
for i in range(1, 6):
    for j in range(int(len(refmults[i]))):
        for k in range(9):
            if np.sum(refmults[i][-(j+1):]) > total[i]*ranges[k] and cents[i][k] == 0.0:
                cents[i][k] = -j-1
for i in range(len(refmults[0])):
    for j in range(9):
        if np.sum(refmults[0][:i]) > total[0]*ranges[j] and cents[0][j] == 0.0:
            cents[0][j] = i

# Now to get those lovely b distributions.
distributions = [[], [], [], [], [], []]

# Distribution of b in UrQMD for all centrality ranges.
for i in range(6):
    distributions[i].append(b[1][0:-1])
distributions[0].append(b[0])

# Refmult distributions of b for all centrality ranges.
dummy = [[], [], [], [], []]
for i in range(5):
    dummy[i].append(refVb[i][int(cents[i+1][0])])
dummy = np.asarray(dummy)
for i in range(5):
    for k in range(-int(cents[i+1][0])):
        dummy[i][0] += refVb[i][-k-1]
print(cents)
for i in range(1, 6):
    distributions[i].append(dummy[i-1])

for l in range(1, 9):
    dummy = [[], [], [], [], []]
    for i in range(5):
        dummy[i].append(refVb[i][int(cents[i+1][l])])
    dummy = np.asarray(dummy)
    for i in range(5):
        for k in range(-int(cents[i+1][l]), -int(cents[i+1][l-1])):
            dummy[i][0] += refVb[i][-k-1]
    for i in range(1, 6):
        distributions[i].append(dummy[i-1])
distributions = np.asarray(distributions)

# Now to get the variances.
# Values are as [[b],[refmult1],[refmult2],[Fwd1],[Fwd2],[Fwd3]].
# In each [], there are the n centrality ranges.

# All the refmults and corresponding b values
bVals = [[], [], [], [], [], []]

# Distributions of b values in each centrality range
bDist = [[], [], [], [], [], []]

# Variances for b distributions
bVars = [[], [], [], [], [], []]

# Scaled variances
bVarsPhi = [[], [], [], [], [], []]

for i in range(6):
    for j in range(9):
        bDist[i].append([])
        bVars[i].append([])
        bVals[i].append([])
        bVarsPhi[i].append([])

#--------------------bVals------------------------------------#
# b values and corresponding amounts in each centrality range
# for UrQMD's b, 0-5%:
bVals[0][0].append(distributions[0][0][:int(cents[0][0])])
bVals[0][0].append(distributions[0][1][:int(cents[0][0])])
for i in range(1, 9):
    bVals[0][i].append(distributions[0][0]
                       [int(cents[0][i-1]):int(cents[0][i])])
    bVals[0][i].append(distributions[0][1]
                       [int(cents[0][i-1]):int(cents[0][i])])
# All other centrality ranges:
for k in range(9):
    for i in range(int(len(bVals[0][k][0]))):
        if bVals[0][k][1][i] != 0:
            for j in range(int(bVals[0][k][1][i])):
                bDist[0][k].append(bVals[0][k][0][i])

# Now for all the refmults.
for i in range(1, 6):
    for j in range(9):
        bVals[i][j].append(distributions[i][0])
        bVals[i][j].append(distributions[i][j+1][0])

for i in range(1, 6):
    for j in range(9):
        for k in range(int(len(bVals[i][j][0]))):
            if bVals[i][j][1][k] != 0:
                for l in range(int(bVals[i][j][1][k])):
                    bDist[i][j].append(bVals[i][j][0][k])
bDist = np.asarray(bDist)
for j in range(9):
    print("Vars for all b-Fwd3 in range", j, ":")
    for i in range(6):
        print(np.var(bDist[i][j]))
for i in range(6):
    for j in range(9):
        bVars[i][j].append(np.var(bDist[i][j]))
print(bVars[0])
for i in range(1, 6):
    for j in range(9):
        bVarsPhi[i][j] = bVars[i][j][0]/bVars[0][j][0]

#+++++++++++++++++++++++++#
#:::::::::::::::::::::::::#
# Time to make some plots #
#:::::::::::::::::::::::::#
#+++++++++++++++++++++++++#

# Plot of the refmults.
legend = ['refmult1', 'refmult2', 'Fwd1', 'Fwd2', 'Fwd3']
plt.figure(figsize=(16, 9))
plt.plot(refmult1[1][: -1], refmult1[0][:] /
         np.max(refmult1[0]), label='refmult1', linewidth=3.0)
plt.plot(refmult2[1][: -1], refmult2[0][:] /
         np.max(refmult2[0]), label='refmult2', linewidth=3.0)
plt.plot(Fwd1[1][: -1], Fwd1[0][:]/np.max(Fwd1[0]),
         label='Fwd1', linewidth=3.0)
plt.plot(Fwd2[1][: -1], Fwd2[0][:]/np.max(Fwd2[0]),
         label='Fwd2', linewidth=3.0)
plt.plot(Fwd3[1][: -1], Fwd3[0][:]/np.max(Fwd3[0]),
         label='Fwd3', linewidth=3.0)
plt.xlim(0, 500)
# plt.ylim(0.01, 10.01)
plt.legend(fontsize=20)
plt.title(r'$N_{ch}$ in Different $\eta$ Bins', fontsize=30)
plt.xlabel('$N_{ch}$', fontsize=20)
plt.ylabel('nCount', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.yscale('log')
plt.show()
plt.close()

Title1 = ['0-5%', '5-10%', '10-20%', '20-30%',
          '30-40%', '40-50%', '50-60%', '60-70%', '70-80%']
legend1 = ['b', 'refmult1', 'refmult2', 'Fwd1', 'Fwd2', 'Fwd3']
# Plot the b distributions.
fig, axes = plt.subplots(nrows=3, ncols=3)
axes[0, 0].semilogy(distributions[0][0][:int(cents[0][0])],
                    distributions[0][1][:int(
                        cents[0][0])]/np.sum(distributions[0][1][:int(cents[0][0])]),
                    '-b')

for i in range(3):
    for j in range(3):
        x = i*3 + j
        if x != 0:
            axes[i, j].semilogy(distributions[0][0][int(cents[0][x-1]):int(cents[0][x])],
                                distributions[0][1][int(cents[0][x-1]):int(cents[0][x])] /
                                np.sum(distributions[0][1][int(cents[0][x-1]):int(cents[0][x])]))

for i in range(3):
    for j in range(3):
        x = i*3 + j
        for k in range(1, 6):
            axes[i, j].semilogy(distributions[k][0], distributions[k]
                                [x+1][0]/np.sum(distributions[k][x+1][0]))
            axes[i, j].title.set_text(Title1[x])
leg = plt.legend(legend1)
for line in leg.get_lines():
    line.set_linewidth(4.0)
plt.show()
plt.close()

x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
xLabels = []
for i in range(int(len(Title1))):
    xLabels.append(Title1[-i-1])
legend2 = legend1[1:]

y = [[], [], [], [], []]
for i in range(5):
    for j in range(9):
        y[i].append(bVarsPhi[i+1][-j-1])

fig, ax = plt.subplots()
s = 250
plt.scatter(x, y[0], s=s, facecolors='none', edgecolors='blue')
plt.scatter(x, y[1], marker='s', s=s, facecolors='none', edgecolors='black')
plt.scatter(x, y[2], marker='<', s=s,
            facecolors='none', edgecolors='purple')
plt.scatter(x, y[3], marker='^', s=s, facecolors='none', edgecolors='red')
plt.scatter(x, y[4], marker='p', s=s, facecolors='none', edgecolors='blue')
plt.xticks(x, xLabels, rotation=60)
# ax.set_xticklabels(xLabels, rotation=60)
leg = plt.legend(legend2, fontsize=15)
plt.title(r'$\Phi$ for 19.6 GeV', fontsize=30)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Centrality class (%)', fontsize=20)
plt.yscale('log')
plt.ylabel(
    r'$\Phi$ = $\sigma^{2}_{b_{X}}$/$\sigma^{2}_{b_{centrality-b}}$', fontsize=20)
plt.show()
