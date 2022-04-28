import numpy as np
import os
import uproot as up
import awkward as ak
import matplotlib.pyplot as plt
import pandas as pd

os.chdir(r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results\BigList")
files = os.listdir()
roots = []
for i in range(len(files)):
    if files[i].endswith(".root"):
        roots.append(files[i])
roots = np.array(roots)
names = []

ref_arr = []
pro_arr = []
anti_arr = []
net_arr = []
ave_std = np.zeros((4, 2, len(roots)))
for i in range(len(roots)):
    data = up.open(roots[i])['Protons']
    for j in range(len(data.keys())):
        if i == 0:
            names.append(data.keys()[j])
        arr = data[data.keys()[j]].array(library='np')
        ave_std[j][0][i] = np.mean(arr)
        ave_std[j][1][i] = np.std(arr) / np.sqrt(len(arr))
        if j == 0:
            ref_arr.append(arr)
        elif j == 1:
            pro_arr.append(arr)
        elif j == 2:
            anti_arr.append(arr)
        elif j == 3:
            net_arr.append(arr)
ref_arr = np.hstack(np.array(ref_arr))
pro_arr = np.hstack(np.array(pro_arr))
anti_arr = np.hstack(np.array(anti_arr))
net_arr = np.hstack(np.array(net_arr))

ref_int = np.unique(ref_arr)
means = []
for i in ref_int:
    index = ref_arr == i
    means.append(np.mean(net_arr[index]))
plt.plot(means)
plt.show()

high_low_ave = [[], [], [], []]
bad_list = []
for i in range(4):
    high_low_ave[i].append(np.mean(ave_std[i][0]))
    high_low_ave[i].append(np.std(ave_std[i][0]))
    if i > 0:
        index = (((ave_std[i][0] + ave_std[i][1]) > (high_low_ave[i][0] + (3 * high_low_ave[i][1]))) |
                 ((ave_std[i][0] - ave_std[i][1]) < (high_low_ave[i][0] - (3 * high_low_ave[i][1]))))
        bad_list.append(roots[index])
bad_list = np.unique(np.hstack(np.array(bad_list)))
bad_list_index = []
ref_cut = []
net_cut = []
for i in range(len(roots)):
    if roots[i] in bad_list:
        bad_list_index.append(True)
    else:
        data = up.open(roots[i])['Protons']
        arr_ref = data['RefMult3'].array(library='np')
        arr_net = data['net_protons'].array(library='np')
        ref_cut.append(arr_ref)
        net_cut.append(arr_net)
        bad_list_index.append(False)
ref_cut = np.hstack(np.array(ref_cut))
net_cut = np.hstack(np.array(net_cut))
ref_cut_int = np.unique(ref_cut)
means_cut = []
for i in ref_cut_int:
    index = ref_cut == i
    means_cut.append(np.mean(net_cut[index]))
plt.plot(means_cut)
plt.show()

fig, ax = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
for i in range(3):
    ax[i].errorbar(roots, ave_std[i+1][0], yerr=ave_std[i+1][1], fmt='ok', ms=0, mfc='None', capsize=1.5,
                      elinewidth=1)
    ax[i].errorbar(roots[bad_list_index], ave_std[i+1][0][bad_list_index],
                   yerr=ave_std[i+1][1][bad_list_index], fmt='ok', ms=0, mfc='None', capsize=1.5,
                      elinewidth=1, ecolor='red', label='bad runs')
    ax[i].set_xticks(ax[i].get_xticks()[::100])
    ax[i].tick_params(labelrotation=45, labelsize=7)
    ax[i].axhline(high_low_ave[i+1][0], 0, 1, c='green', label="Mean")
    ax[i].axhline(high_low_ave[i+1][0] + (3*high_low_ave[i+1][1]), 0, 1, c='blue', label=r'3$\sigma$')
    ax[i].axhline(high_low_ave[i+1][0] - (3*high_low_ave[i+1][1]), 0, 1, c='blue')
    ax[i].set_title(names[i+1], fontsize=20)
    ax[i].legend()
plt.show()
