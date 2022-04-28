import uproot as up
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import colors as mcolors
from matplotlib import cm
# from keras.models import load_model

# Centrality ranges, as follows:
# 90-100%, 80-90%, 70-80%, 60-70%, 50-60%,
# 40-50%, 30-40%, 20-30%, 10-20%, 5-10%, 0-5%
ranges = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
          0.7, 0.8, 0.9]
ranges1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
ranges = np.asarray(ranges)
ranges1 = np.asarray(ranges1)

# Import the arrays from the paper.
energy = str(200)
os.chdir(r'D:\UrQMD_cent_sim\PaperFiles\En_%s\e_by_e' % energy)
files = up.open('ppkset_rap_%s.root' % energy)  # For my data.
urqmd = files['AMPT_tree']
os.chdir(r"D:\UrQMD_cent_sim\%s" % energy)

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

for i in range(r):
    print(np.max(refMult[i]))

centRanges = np.zeros((r, 10))
for i in range(int(len(ranges))):
    centRanges[0][i] = np.quantile(refMult[0], ranges[i])

for i in range(int(len(ranges1))):
    for j in range(1, r):
        centRanges[j][i] = np.quantile(refMult[j], ranges1[i])

print(centRanges)

# b centrality ranges from the paper:
# 7.7 GeV
if energy == '7':
    centNew = np.asarray([[0, 3.3, 4.64, 6.6, 8.1, 9.25, 10.4, 11.4, 12.3, 13.2],
                          [2, 4, 9, 19, 35, 60, 96, 150, 188, 265],
                          [2, 4, 9, 18, 33, 57, 90, 140, 174, 260],
                          [2, 7, 16, 32, 59, 100, 159, 247, 308, 450],
                          [2, 5, 10, 18, 30, 44, 60, 75, 83, 110],
                          [54, 65, 72, 76, 79, 82, 85, 88, 91, 110],
                          [26, 36, 45, 52, 57, 61, 65, 69, 72, 88],
                          [139, 144, 147, 148, 150, 151, 153, 154, 155, 182]])
# 19.6 GeV
if energy == '19':
    centNew = np.asarray([[0, 3.28, 4.62, 6.55, 8.01, 9.28, 10.4, 11.3, 12.4, 13.4],
                          [2, 6, 14, 29, 55, 95, 154, 243, 306, 410],
                          [2, 6, 14, 29, 54, 93, 150, 235, 295, 400],
                          [3, 11, 26, 55, 103, 177, 286, 450, 566, 800],
                          [2, 6, 14, 28, 50, 80, 122, 177, 212, 300],
                          [2, 5, 11, 20, 33, 47, 62, 77, 85, 120],
                          [53, 65, 74, 79, 82, 86, 89, 92, 95, 120],
                          [103, 109, 119, 138, 164, 198, 241, 294, 324, 425]])
# 200 GeV
if energy == '200':
    centNew = np.asarray([[0, 3.31, 4.66, 6.58, 8.05, 9.33, 10.5, 11.4, 12.3, 13.2],
                          [4, 12, 26, 53, 98, 168, 274, 433, 549, 825],
                          [5, 12, 27, 55, 101, 173, 281, 444, 561, 820],
                          [8, 23, 51, 104, 193, 331, 539, 855, 1083, 1600],
                          [8, 20, 44, 87, 156, 259, 405, 612, 752, 1050],
                          [6, 16, 34, 67, 118, 192, 296, 439, 536, 750],
                          [4, 9, 20, 38, 65, 105, 158, 228, 274, 390],
                          [17, 44, 99, 195, 345, 565, 871, 1299, 1581, 1900]])

print(centNew)
np.save('centRanges.npy', centRanges)
np.save('centNew.npy', centNew)

# Distributions of b in each centrality range for each RefMult.
bDists = []
bDistsP = []
for i in range(r):
    bDists.append([])
    bDistsP.append([])
    for j in range(11):
        bDists[i].append([])
        bDistsP[i].append([])

for i in range(int(len(refMult[0]))):
    if refMult[0][i] <= centRanges[0][0]:
        bDists[0][0].append(refMult[0][i])
    for j in range(1, 10):
        if refMult[0][i] > centRanges[0][j-1] and refMult[0][i] <= centRanges[0][j]:
            bDists[0][j].append(refMult[0][i])
        if refMult[0][i] >= centNew[0][j-1] and refMult[0][i] < centNew[0][j]:
            bDistsP[0][j-1].append(refMult[0][i])
    if refMult[0][i] > centRanges[0][9]:
        bDists[0][10].append(refMult[0][i])
    if refMult[0][i] > centNew[0][9]:
        bDistsP[0][9].append(refMult[0][i])
        bDistsP[0][10].append(refMult[0][i])


for i in range(int(len(refMult[0]))):
    if refMult[r-1][i] <= centRanges[r-1][0]:
        bDists[r-1][0].append(refMult[0][i])
    for j in range(1, 10):
        if refMult[r-1][i] > centRanges[r-1][j-1] and refMult[r-1][i] <= centRanges[r-1][j]:
            bDists[r-1][j].append(refMult[0][i])
    if refMult[r-1][i] > centRanges[r-1][9]:
        #    if refMult[r-1][i] > 15.08:
        bDists[r-1][10].append(refMult[0][i])

for i in range(1, r-1):
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

refMult = np.asarray(refMult)
# np.save('refMult.npy', refMult)
# np.save('bDists.npy', bDists)
# np.save('bDistsP.npy', bDistsP)
# np.save('bVars.npy', bVars)
# np.save('bVarsP.npy', bVarsP)
