#
# \Read mean and variance for quantities from pico based Numpy arrays.
#
# \author Skipper Kagamaster
# \date 05/06/2022
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
# /

"""
Parent macro: find_aves.py
This takes the output from the parent and finds bad runs based
on average values for various quantities. Analysis can be done
then on the QA'd pico set.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir(r'C:\200\PythonArrays')
aves = np.load('ave_aves.npy', allow_pickle=True).T
stds = np.load('ave_stds.npy', allow_pickle=True).T
runs = np.load('ave_runs.npy', allow_pickle=True)

# Let's make a Pandas dataframe.
'''We'll do this later.
iterables = [runs, ["ave", "std"]]
pd_index = pd.MultiIndex.from_product(iterables, names=["RunID", "Stats"])
columns = ['<RefMult3>', '<v_z>', '<v_r>', 'zdcx']
for i in range(32):
    columns.append('<ring_{}>'.format(i+1))

df = pd.DataFrame(index=pd_index)
'''
columns = ['<RefMult3>', r'<$v_z$>', r'<$v_r$>', r'<$ZDC_x$>']
for i in range(32):
    columns.append(r'<ring_{}>'.format(i+1))
columns.extend([r'<$p_t$>', r'<$\phi$>', r'<$\eta$>', '<DCA>'])
set1 = [0, 1, 2, 3, 36, 37, 38, 39]
set2 = np.linspace(4, 35, 32).astype('int')

for i in range(40):
    if i < 36:
        stds[i] = np.divide(stds[i], np.sqrt(runs[1]))
    else:
        stds[i] = np.divide(stds[i], np.sqrt(runs[2]))

ave_ave = np.mean(aves, axis=1)
std_ave = np.std(aves, axis=1)
bad_runs = []
for i in range(len(aves)):
    arr = (aves[i] > (ave_ave[i] + std_ave[i])) | (aves[i] < (ave_ave[i] - std_ave[i]))
    run_arr = runs[0][arr]
    bad_runs.append(run_arr)
bad_runs = np.unique(np.hstack(bad_runs))
np.save('badruns.npy', bad_runs)
index_up = []
for i in range(len(runs[0])):
    if runs[0][i] in bad_runs:
        index_up.append(i)
print("Runs marked bad:", np.round(len(index_up)/len(runs[0])*100, 2), "%")
print("Good events:", np.sum(runs[1]) - np.sum(runs[1][index_up]))

fig, ax = plt.subplots(2, 4, figsize=(16, 9), constrained_layout=True)
for i in range(2):
    for j in range(4):
        x = i*4 + j
        rid = np.linspace(0, len(runs[0])-1, len(runs[0]))
        ax[i, j].errorbar(rid, aves[set1[x]], yerr=stds[set1[x]], fmt='ok', ms=1,
                          mfc='None', elinewidth=0.1,
                          label="Good Runs")
        ax[i, j].errorbar(rid[index_up], aves[set1[x]][index_up],
                          yerr=stds[set1[x]][index_up], marker='o', color='orange', ms=2,
                          mfc='None', elinewidth=0.2, lw=0,
                          label="Bad Runs", ecolor='orange')
        ax[i, j].axhline(ave_ave[set1[x]], c='r', ls='--', label=r"$\mu$")
        ax[i, j].axhline(ave_ave[set1[x]]+std_ave[set1[x]], c='b', ls='--', label=r"$\sigma$")
        ax[i, j].axhline(ave_ave[set1[x]]-std_ave[set1[x]], c='b', ls='--')
        ax[i, j].set_ylim(ave_ave[set1[x]]-3*std_ave[set1[x]],
                            ave_ave[set1[x]]+3*std_ave[set1[x]])
        ax[i, j].set_ylabel(columns[set1[x]], fontsize=15)
        ax[i, j].set_xlabel("RunID", loc='right', fontsize=15)
ax[1, 0].legend(fontsize=15)
plt.show()

fig, ax = plt.subplots(4, 4, figsize=(16, 9), constrained_layout=True)
for i in range(4):
    for j in range(4):
        x = i*4 + j
        rid = np.linspace(0, len(runs[0])-1, len(runs[0]))
        ax[i, j].errorbar(rid, aves[set2[x]], yerr=stds[set2[x]], fmt='ok', ms=1,
                          mfc='None', elinewidth=0.1,
                          label="Good Runs")
        ax[i, j].errorbar(rid[index_up], aves[set2[x]][index_up],
                          yerr=stds[set2[x]][index_up], marker='o', ms=2,
                          mfc='None', elinewidth=0.2, color='orange',
                          label="Bad Runs", ecolor='orange', lw=0)
        ax[i, j].axhline(ave_ave[set2[x]], c='r', ls='--', label=r"$\mu$")
        ax[i, j].axhline(ave_ave[set2[x]]+std_ave[set2[x]], c='b', ls='--', label=r"$\sigma$")
        ax[i, j].axhline(ave_ave[set2[x]]-std_ave[set2[x]], c='b', ls='--')
        ax[i, j].set_ylim(ave_ave[set2[x]]-3*std_ave[set2[x]],
                            ave_ave[set2[x]]+3*std_ave[set2[x]])
        ax[i, j].set_ylabel(columns[set2[x]], fontsize=15)
        ax[i, j].set_xlabel("RunID", loc='right', fontsize=15)
ax[0, 0].legend(fontsize=15)
plt.show()

fig, ax = plt.subplots(4, 4, figsize=(16, 9), constrained_layout=True)
for i in range(4):
    for j in range(4):
        x = i*4 + j + 16
        rid = np.linspace(0, len(runs[0])-1, len(runs[0]))
        ax[i, j].errorbar(rid, aves[set2[x]], yerr=stds[set2[x]], fmt='ok', ms=1,
                          mfc='None', elinewidth=0.1)
        ax[i, j].errorbar(rid[index_up], aves[set2[x]][index_up],
                          yerr=stds[set2[x]][index_up], marker='o', ms=2,
                          mfc='None', elinewidth=0.2, color='orange',
                          label="Bad Runs", ecolor='orange', lw=0)
        ax[i, j].axhline(ave_ave[set2[x]], c='r', ls='--', label=r"$\mu$")
        ax[i, j].axhline(ave_ave[set2[x]]+std_ave[set2[x]], c='b', ls='--', label=r"$\sigma$")
        ax[i, j].axhline(ave_ave[set2[x]]-std_ave[set2[x]], c='b', ls='--')
        ax[i, j].set_ylim(ave_ave[set2[x]]-3*std_ave[set2[x]],
                            ave_ave[set2[x]]+3*std_ave[set2[x]])
        ax[i, j].set_ylabel(columns[set2[x]], fontsize=15)
        ax[i, j].set_xlabel("RunID", loc='right', fontsize=15)
ax[0, 0].legend(fontsize=15)
plt.show()
