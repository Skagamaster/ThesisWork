import numpy as np
import os
import matplotlib.pyplot as plt
import uproot as up
import awkward as ak
from scipy.stats import skew, kurtosis

os.chdir(r"C:\Users\dansk\Documents\Thesis\Protons\WIP\Protons")

protons_net = np.load("proton_sum.npy", allow_pickle=True)
for i in range(20):
    print(len(protons_net[i]))

cbwc = [[], [], [], []]
no_cbwc = [[], [], [], []]
raw = np.zeros((4, 1000))
lens = []
RefCuts = np.asarray((0, 10, 21, 41, 72, 118, 182, 270, 392, 472))
for i in range(1000):
    try:
        ind = protons_net[i]
        raw[0][i] = np.mean(ind)
        raw[1][i] = np.var(ind)
        raw[2][i] = skew(ind) * np.power(np.sqrt(np.var(ind)), 3)
        raw[3][i] = kurtosis(ind) * np.power(np.var(ind), 2)
    except Exception as e:
        continue

protons_net = ak.Array(protons_net)
print(protons_net)
for i in range(len(RefCuts)-1):
    ind = ak.flatten((protons_net[RefCuts[i]:RefCuts[i+1]]), axis=None)
    lens.append(len(ind))
    no_cbwc[0].append(np.mean(ind))
    no_cbwc[1].append(np.var(ind))
    no_cbwc[2].append(skew(ind) * np.power(np.sqrt(np.var(ind)), 3))
    no_cbwc[3].append(kurtosis(ind) * np.power(np.var(ind), 2))
ind = ak.flatten((protons_net[RefCuts[-1]:]), axis=None)
lens.append(len(ind))
no_cbwc[0].append(np.mean(ind))
no_cbwc[1].append(np.var(ind))
no_cbwc[2].append(skew(ind) * np.power(np.sqrt(np.var(ind)), 3))
no_cbwc[3].append(kurtosis(ind) * np.power(np.var(ind), 2))

# Let's make some plots!
fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
for i in range(2):
    for j in range(2):
        x = i*2 + j
        ax[i, j].plot(no_cbwc[x])
plt.show()

fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
for i in range(2):
    for j in range(2):
        x = i*2 + j
        ax[i, j].plot(raw[x])
plt.show()
