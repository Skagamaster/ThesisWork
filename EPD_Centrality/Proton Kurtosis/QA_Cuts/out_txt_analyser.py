import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import uproot as up
import awkward as ak
from scipy.stats import skew, kurtosis

# Load Yu's data.
yu_params = up.open(r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Fast14.5_Result\Cumulant_NotCorrected.root")
yu_cumulants_int = [yu_params["NetC1"].values(),
                    yu_params["NetC2"].values(),
                    yu_params["NetC3"].values(),
                    yu_params["NetC4"].values()]
yu_x = np.linspace(0, len(yu_cumulants_int)-1, len(yu_cumulants_int))

os.chdir(r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results\BigList")
high_pro_aves = np.load('high_pro_aves.npy', allow_pickle=True)
refx = np.load('refx.npy', allow_pickle=True)
files = []
for i in os.listdir():
    if os.path.getsize(i) <= 3e6:
        continue
    if (i.endswith(".txt")) & (i != "out_all.txt") & i.startswith("out"):
        files.append(i)
# For testing.
# files = files[:100]

high_aves = []
high_runs = []
pro_ref_ints = []
pro_mean = np.zeros((len(files), len(refx)))

count = 0
print("Working on file:")
for i in range(len(files)):
    if count % 100 == 0:
        print(count, "of", len(files))
    try:
        data = np.loadtxt(files[i])
        pro_ref_ints.append([])
        netp = data[:, 4]
        refp = data[:, 1]
        mean = np.mean(netp)
        high_aves.append(mean)
        high_runs.append(data[0][0])
        for j in range(len(refx)):
            index = refp == refx[j]
            pro_ref_ints[i].append(netp[index])
            if len(netp[index]) > 0:
                pro_mean[i][j] = np.mean(netp[index])
        count += 1
    except Exception as e:
        count += 1
        print("File", files[i], "is no good; kill that fool!")
full_mean = np.mean(pro_mean, axis=1)
pro_refs = ak.Array(pro_ref_ints)
pro_ref_ints = np.array(pro_ref_ints)
plt.pcolormesh(pro_mean, cmap='jet')
plt.colorbar(label=r'<Net Proton>')
plt.title(r"RefMult3 vs <Net Proton>", fontsize=30)
plt.xlabel("RefMult3", fontsize=20)
plt.ylabel("Run ID", fontsize=20)
plt.show()

pro_means_int = np.zeros((4, len(refx)))
lens = np.zeros(len(refx))
for i in range(len(refx)):
    arr = np.hstack(pro_ref_ints[:, i])
    lens[i] = len(arr)
    pro_means_int[0][i] = np.mean(arr)
    pro_means_int[1][i] = np.var(arr)
    pro_means_int[2][i] = (skew(arr) * np.power(np.sqrt(np.var(arr)), 3))
    pro_means_int[3][i] = (kurtosis(arr) * np.power(np.var(arr), 2))

fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
y_labels = [r"$C_1$", r"$C_2$", r"$C_3$", r"$C_4$"]
for i in range(2):
    for j in range(2):
        x = i*2 + j
        ax[i, j].plot(pro_means_int[x])
        ax[i, j].plot(yu_cumulants_int[x], c='r', alpha=0.5)
        ax[i, j].set_ylabel(y_labels[x], fontsize=20)
        ax[i, j].set_xlabel("RefMult3", fontsize=20)
plt.suptitle(r"Cumulants for Net Proton by RefMult3", fontsize=30)
plt.show()

np.save('high_pro_aves.npy', np.array((high_runs, high_aves)))
mean = np.mean(high_aves)
std = 3*np.std(high_aves)

plt.scatter(high_runs, high_aves, s=5, c='black')
plt.axhline(mean, 0, 1, c='r', label=r"$\mu$")
plt.axhline(mean-std, 0, 1, c='b', label=r"$3\sigma$")
plt.axhline(mean+std, 0, 1, c='b')
plt.title(r"<Net Proton> per Run", fontsize=30)
plt.ylabel(r"<Net Proton>", fontsize=20)
plt.xlabel("Run ID", fontsize=20)
plt.legend()
plt.show()
