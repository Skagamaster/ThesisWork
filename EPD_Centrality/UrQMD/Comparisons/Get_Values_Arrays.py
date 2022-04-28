import numpy as np
import uproot as up
import matplotlib.pyplot as plt
import os

# Input the energy as in the folder (integer numbers).
energy = str(19)
col = energy
if energy == '19':
    col = '19.6'
if energy == '7':
    col = '7.7'

# Centrality ranges, as follows:
# 90-100%, 80-90%, 70-80%, 60-70%, 50-60%,
# 40-50%, 30-40%, 20-30%, 10-20%, 5-10%, 0-5%
ranges = np.asarray([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
ranges1 = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
ranLen = int(len(ranges))

# Import the arrays from the paper.
os.chdir(r'D:\UrQMD_cent_sim\PaperFiles\En_%s\e_by_e' % energy)
files = up.open('ppkset_rap_%s.root' % energy)
urqmd = files['AMPT_tree']
# I am excluding branches 1-4
ul = np.hstack(('Imp', urqmd.keys()[5:-1]))
os.chdir(r'D:\UrQMD_cent_sim')

refMult = []
for i in range(int(len(ul))):
    refMult.append([])
#refMult[0] = urqmd.array('Imp')
index = 0
for i in ul:
    refMult[index] = urqmd.array(i)
    index += 1
r = int(len(refMult))

# Now to get centrality ranges.
centRanges = np.zeros((r, ranLen))

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
    for j in range(ranLen+1):
        bDists[i].append([])

for i in range(int(len(refMult[0]))):
    if refMult[0][i] <= centRanges[0][0]:
        bDists[0][0].append(refMult[0][i])
    for j in range(1, ranLen):
        if refMult[0][i] > centRanges[0][j-1] and refMult[0][i] <= centRanges[0][j]:
            bDists[0][j].append(refMult[0][i])
    if refMult[0][i] > centRanges[0][-1]:
        bDists[0][-1].append(refMult[0][i])

for i in range(1, r):
    for j in range(int(len(refMult[0]))):
        if refMult[i][j] <= centRanges[i][0]:
            bDists[i][-1].append(refMult[0][j])
        for k in range(0, ranLen):
            if refMult[i][j] > centRanges[i][k-1] and refMult[i][j] <= centRanges[i][k]:
                bDists[i][-k-1].append(refMult[0][j])
        if refMult[i][j] > centRanges[i][-1]:
            bDists[i][0].append(refMult[0][j])
bDists = np.asarray(bDists)

bVars = np.empty((r, ranLen+1))
for i in range(ranLen+1):
    for j in range(r):
        bVars[j][i] = np.var(bDists[j][-i-1])/np.var(bDists[0][-i-1])
print(bVars)

outFile = 'bVars%s.txt' % energy
outFile1 = 'bDists%s.txt' % energy
np.savetxt(outFile, bVars[1:])
np.savetxt(outFile1, bDists)
