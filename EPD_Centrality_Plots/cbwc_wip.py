import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import json

labels = [r"$C_1$", r"$C_2$", r"$C_3$", r"$C_4$"]
titles = ["RefMult3", r"$\xi_{Linear}$", r"$\xi_{ReLU}$", r"$\xi_{Swish}$"]
ranges = ["80-100%", "70-80%", "60-70%", "50-60%", "40-50%",
          "30-40%", "20-30%", "10-20%", "5-10%", "0-5%"]
marker = ["*", "o", "^", "P"]
RefCuts = [-999, 10, 21, 41, 72, 118, 182, 270, 392, 472]  # Yu's determination of RefMult3 centrality via Glauber.

with open(r'C:\Users\dansk\Documents\Thesis\2021_DNP_Work\arrays\protons.txt', 'r') as f:
    protons = json.load(f)
with open(r'C:\Users\dansk\Documents\Thesis\2021_DNP_Work\arrays\n_arr.txt', 'r') as f:
    n_arr = json.load(f)
with open(r'C:\Users\dansk\Documents\Thesis\2021_DNP_Work\arrays\x_vals.txt', 'r') as h:
    x_vals = json.load(h)
quants = np.load(r'C:\Users\dansk\Documents\Thesis\2021_DNP_Work\arrays\quants.npy', allow_pickle=True)
quant_arr = []
for i in range(len(quants)):
    quant_arr.append(np.hstack(([-999], quants[i])))
quants = np.array(quant_arr)
print(quants)
protons = ak.Array(protons)
n_arr = ak.Array(n_arr)
x_vals = ak.Array(x_vals)

# Plot of the raw cumulants per unique centrality value.
fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
for k in range(4):
    for i in range(2):
        for j in range(2):
            x = i*2 + j
            ax[i, j].plot(x_vals[k], protons[k][x], lw=0, marker=marker[k], ms=3, alpha=0.5, label=titles[k])
            ax[i, j].set_ylabel(labels[x], fontsize=20)
            ax[i, j]. set_xlabel("C (centrality #)", fontsize=15)
ax[1, 1].legend(fontsize=10)
plt.suptitle("Raw Net Proton Cumulants", fontsize=30)
plt.show()

# Without CBWC
prot_bins = []
for k in range(4):
    prot_bins.append([])
    for m in range(4):
        prot_bins[k].append([])
        protest = protons[k][m]
        for i in range(len(quants[k])-1):
            index = ((x_vals[k] >= quants[k][i]) & (x_vals[k] < quants[k][i+1]))
            filler = np.mean(protest[index])
            prot_bins[k][m].append(filler)
        index = (x_vals[k] >= quants[k][-1])
        filler = np.mean(protest[index])
        prot_bins[k][m].append(filler)

fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
for k in range(4):
    for i in range(2):
        for j in range(2):
            x = i*2 + j
            ax[i, j].plot(ranges, prot_bins[k][x], lw=0, marker=marker[k], ms=10, alpha=0.5, label=titles[k])
            ax[i, j].set_ylabel(labels[x], fontsize=20)
            ax[i, j]. set_xlabel("Centrality Range", fontsize=15)
ax[1, 1].legend(fontsize=10)
plt.suptitle("Net Proton Cumulants without CBWC", fontsize=30)
plt.show()

# Applying CBWC
prot_bins_cbwc = []
for k in range(4):
    prot_bins_cbwc.append([])
    for m in range(4):
        prot_bins_cbwc[k].append([])
        protest = protons[k][m] * n_arr[k]
        for i in range(len(quants[k])-1):
            index = ((x_vals[k] >= quants[k][i]) & (x_vals[k] < quants[k][i+1]))
            n_sum = np.sum(n_arr[k][index])
            filler = np.sum(protest[index])/n_sum
            prot_bins_cbwc[k][m].append(filler)
        index = (x_vals[k] >= quants[k][-1])
        n_sum = np.sum(n_arr[k][index])
        filler = np.sum(protest[index])/n_sum
        prot_bins_cbwc[k][m].append(filler)

fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
for k in range(4):
    for i in range(2):
        for j in range(2):
            x = i*2 + j
            ax[i, j].plot(ranges, prot_bins_cbwc[k][x], lw=0, marker=marker[k], ms=10, alpha=0.5, label=titles[k])
            ax[i, j].set_ylabel(labels[x], fontsize=20)
            ax[i, j]. set_xlabel("Centrality Range", fontsize=15)
ax[1, 1].legend(fontsize=10)
plt.suptitle("Net Proton Cumulants with CBWC", fontsize=30)
plt.show()

# And now for the cumulants themselves.
prot_bins_cbwc = np.array(prot_bins_cbwc)
for i in range(4):
    prot_bins_cbwc[i][2] = np.divide(prot_bins_cbwc[i][2], prot_bins_cbwc[i][1])
    prot_bins_cbwc[i][3] = np.divide(prot_bins_cbwc[i][3], prot_bins_cbwc[i][1])
    prot_bins_cbwc[i][1] = np.divide(prot_bins_cbwc[i][1], prot_bins_cbwc[i][0])
titles_comp = [r"$\mu$", r"$\frac{\sigma^2}{\mu}$", r"$S\sigma$", r"$\kappa\sigma^2$"]
fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
for k in range(4):
    for i in range(2):
        for j in range(2):
            x = i*2 + j
            ax[i, j].plot(ranges, prot_bins_cbwc[k][x], lw=0, marker=marker[k], ms=10,
                          alpha=0.5, label=titles[k])
            ax[i, j].set_ylabel(titles_comp[x], fontsize=20)
            ax[i, j]. set_xlabel("Centrality Range", fontsize=15)
ax[1, 1].legend(fontsize=10)
plt.suptitle("Net Proton Cumulants (Ratios)", fontsize=30)
plt.show()

# Comparison plots.
for k in range(4):
    fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    for i in range(2):
        for j in range(2):
            x = i*2 + j
            ax[i, j].plot(ranges, prot_bins[k][x], lw=0, marker="*", ms=10, alpha=0.5, label="no cbwc")
            ax[i, j].plot(ranges, prot_bins_cbwc[k][x], lw=0, marker="P", ms=10, alpha=0.5, label="cbwc")
            ax[i, j].set_ylabel(labels[x], fontsize=20)
            ax[i, j]. set_xlabel("Centrality Range", fontsize=15)
    ax[1, 1].legend(fontsize=10)
    plt.suptitle("Cumulant Comparison, {}".format(titles[k]), fontsize=30)
    plt.show()
    plt.close()
