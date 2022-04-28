import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages

os.chdir(r"C:\Users\dansk\Documents\Thesis\Tristan")

chi2 = np.load("chi2_2_test.npy", allow_pickle=True)
"""
f goes from 0-1 in 11 steps.
k goes from 10 to 100 in 10 steps.
mu goes from 0 to 1 in 11 steps.
"""

range_1 = np.linspace(0, 10, 11).astype('int')
range_2 = np.linspace(0, 9, 10).astype('int')
arr = np.linspace(0, 10, 11)
fig, ax = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
for i in range_1:
    print(i, ":", np.min(chi2[i]))
    x = i % 4
    y = int(i/4)
    im = ax[y, x].pcolormesh(chi2[i], cmap="jet", norm=LogNorm())
    ax[y, x].set_xlabel(r"$\mu$ * 10")
    ax[y, x].set_ylabel(r"$\frac{k}{10}$")
    ax[y, x].set_title("f={}".format(arr[i]))
    fig.colorbar(im, ax=ax[y, x])
ax[2, 3].set_axis_off()
plt.show()
