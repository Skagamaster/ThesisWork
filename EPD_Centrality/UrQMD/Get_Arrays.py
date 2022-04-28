import numpy as np
import os
import uproot as up
import matplotlib.pyplot as plt

energy = 19
directory = r'D:\UrQMD_cent_sim\%s' % energy
os.chdir(directory)
data = up.open('CentralityNtuple.root')
refMult = []
for i in range(44):
    refMult.append([])
    refMult[i] = np.asarray(data["Rings"].arrays(
        [data["Rings"].keys()[i]])[data["Rings"].keys()[i]])
rf = int(len(refMult[0]))
# This is for the NTuple I made without all the rings (as a check).
newmult = []
for i in range(int(len(data['refmults'].keys()))):
    newmult.append([])
    newmult[i] = np.asarray(data['refmults'].arrays(
        data['refmults'].keys()[i])[data['refmults'].keys()[i]])
nf = int(len(newmult[0]))

#----------------------------------------------------------------#
# The above just made an array for all the data we need. It's    #
# arranged as follows: refMult[0-15] = EPD ring nMIP             #
# refMult[16] = TPCMult, refMult[17] = b, refMult[18] = particle #
# refMult[19-21] = refMult1-3, refMult[22-26] = FwdAll, Fwd1-3,  #
# FwdAll-p, refMult[27-43] = nMIPraw, ring particles             #
#----------------------------------------------------------------#

# For now, let's just find the cuts for the main culprits.
# Quantile for b:
ranges = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# Quantile for RefMults (opposite correlations):
ranges1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
rl = int(len(ranges))

centRanges = []
newRanges = []
for i in range(8):
    centRanges.append([])
    newRanges.append([])
cl = int(len(centRanges))
# First, we'll get the ranges for b.
for i in range(rl):
    centRanges[0].append(np.quantile(refMult[17], ranges[i]))
    newRanges[0].append(np.quantile(newmult[0], ranges[i]))
# Ranges for all the multiplicities.
for i in range(1, cl):
    for j in range(rl):
        centRanges[i].append(np.quantile(refMult[i+19], ranges1[j]))
        newRanges[i].append(np.quantile(newmult[i], ranges1[j]))

# Now to get the b distributions.
bDists = []
newDists = []
for i in range(cl):
    bDists.append([])
    newDists.append([])
    for j in range(rl+1):
        bDists[i].append([])
        newDists[i].append([])

for i in range(rf):
    b = refMult[17][i]
    newb = newmult[0][i]
    if b <= centRanges[0][0]:
        bDists[0][0].append(b)
    if newb <= newRanges[0][0]:
        newDists[0][0].append(newb)
    for j in range(1, rl):
        if b > centRanges[0][j-1] and b <= centRanges[0][j]:
            bDists[0][j].append(b)
        if newb > newRanges[0][j-1] and newb <= newRanges[0][j]:
            newDists[0][j].append(newb)
    if b > centRanges[0][rl-1]:
        bDists[0][rl].append(b)
    if newb > newRanges[0][rl-1]:
        newDists.append(newb)
    for k in range(1, cl):
        mult = refMult[k+19][i]
        multnew = newmult[k][i]
        if mult > centRanges[k][rl-1]:
            bDists[k][0].append(b)
        if multnew > newRanges[k][rl-1]:
            newDists[k][0].append(newb)
        if mult <= centRanges[k][0]:
            bDists[k][rl].append(b)
        if multnew <= newRanges[k][0]:
            newDists[k][rl].append(newb)
        for l in range(1, rl):
            if mult > centRanges[k][l-1] and mult <= centRanges[k][l]:
                bDists[k][-l-1].append(b)
            if multnew > newRanges[k][l-1] and multnew <= newRanges[k][l]:
                newDists[k][-l-1].append(newb)

bDists = np.asarray(bDists)
newDists = np.asarray(newDists)

# Now to get the variances.
bVars = []
bVarsNew = []
for i in range(cl):
    bVars.append([])
    bVarsNew.append([])
    for j in range(rl+1):
        bVars[i].append(np.var(bDists[i][j])/np.var(bDists[0][j]))
        bVarsNew[i].append(np.var(newDists[i][j])/np.var(newDists[0][j]))
bVars = np.asarray(bVars)
bVarsNew = np.asarray(bVarsNew)

print(centRanges)

#np.save('centRangesBill.npy', centRanges)
#np.save('bDistsBill.npy', bDists)
#np.save('bVarsBill.npy', bVars)
#np.save('bVarsNew.npy', bVarsNew)
