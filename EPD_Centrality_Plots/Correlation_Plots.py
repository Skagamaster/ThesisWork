# \author Skipper Kagamaster
# \date 06/24/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
"""
The purpose of this code is to make plots for the correlation/goodness of fit
for different EPD centrality measures and RefMult3.
"""

import uproot as up
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import skew, kurtosis

# Choose the activation function you want to compare.
actFunc = 'linear'
# actFunc = "relu"
# actFunc = "swish"
# actFunc = "mish"
# actFunc = "bose"

# actFunc1 = 'linear'
# actFunc1 = "relu"
actFunc1 = "swish"
# actFunc1 = "mish"
# actFunc1 = "bose"

proton_params = pd.read_pickle(r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results\BigList\out_all.pkl")
ref_data = proton_params['refmult'].to_numpy()
ref_data[ref_data == 0] = 0.1  # To avoid mathematical infinities.

pred_LW = np.load(r'C:\Users\dansk\Documents\Thesis\2021_DNP_Work\ML_fits\{}predictions.npy'.format(actFunc), allow_pickle=True)
pred_ML = np.load(r'C:\Users\dansk\Documents\Thesis\2021_DNP_Work\ML_fits\{}predictions.npy'.format(actFunc1), allow_pickle=True)
ratio_LW = np.divide(abs(np.subtract(pred_LW, ref_data)), ref_data)
ratio_ML = np.divide(abs(np.subtract(pred_ML, ref_data)), ref_data)
print(np.mean(ratio_LW), np.var(ratio_LW), skew(ratio_LW), kurtosis(ratio_LW))
print(np.mean(ratio_ML), np.var(ratio_ML), skew(ratio_ML), kurtosis(ratio_ML))

fig, ax = plt.subplots(1, figsize=(16, 9), constrained_layout=True)

# ax.hist(ratio_LW, bins=100, label=r"$\xi_{LW}$", color='b', histtype='step')
# ax.hist(ratio_ML, bins=100, label=r"$\xi_{ML}$", color='r', histtype='step')
ax.hist2d(ref_data, ratio_LW, label="LW", bins=200, cmin=1)
# ax.set_yscale('log')
ax.set_title(r"Absolute $\Delta$ between $\xi_x$ and RefMult3", fontsize=30)
ax.set_xlabel(r"$|\Delta|$", fontsize=20)
ax.set_ylabel("Counts", fontsize=20)
ax.legend()
plt.show()

fig, ax = plt.subplots(1, figsize=(16, 9), constrained_layout=True)
ax.scatter(ref_data, ratio_ML, label="ML")
ax.set_title(r"Absolute $\Delta$ between $\xi_x$ and RefMult3", fontsize=30)
ax.set_xlabel(r"$|\Delta|$", fontsize=20)
ax.set_ylabel("Counts", fontsize=20)
ax.legend()
plt.show()
