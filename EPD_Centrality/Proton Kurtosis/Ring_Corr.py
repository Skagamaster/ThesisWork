import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot as up
import os
import functions as fn
import seaborn as sns
from pandas.plotting import scatter_matrix as scx

os.chdir(r'D:\UrQMD_cent_sim\7')
data = up.open('CentralityNtuple.root')
rings = []
for i in range(16):
    rings.append(np.asarray(data["Rings"].arrays(
        [data["Rings"].keys()[i]])[data["Rings"].keys()[i]]))

rings = np.asarray(rings)

b = np.asarray(data["Rings"].arrays(
    [data["Rings"].keys()[17]])[data["Rings"].keys()[17]])

df = pd.DataFrame(rings.T, columns=["r01", "r02", "r03", "r04", "r05", "r06",
                                    "r07", "r08", "r09", "r10", "r11", "r12",
                                    "r13", "r14", "r15", "r16"])

#scx(df, alpha=0.2, figsize=(9, 9), diagonal='kde')
#sns.pairplot(df, kind="kde")

fig, ax = plt.subplots(16, 16, figsize=(10, 9))
fig.subplots_adjust(hspace=0)
for i in range(16):
    for j in range(16):
        ax[i, j].hist2d(rings[i], rings[j], bins=50, cmin=50, cmap="jet")
        ax[i, j].set_axis_off()

plt.tight_layout()
plt.show()
plt.close()

fig, ax = plt.subplots(4, 4, figsize=(10, 9))
fig.subplots_adjust(hspace=0)
for i in range(4):
    for j in range(4):
        x = i*4+j
        ax[i, j].hist2d(b, rings[x], bins=100, cmin=10, cmap="jet")
        ax[i, j].set_axis_off()

plt.tight_layout()
plt.show()
plt.close()
