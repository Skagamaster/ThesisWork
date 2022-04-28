# \author Skipper Kagamaster
# \date 06/24/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
"""
The purpose of this code is to make display graphs for RefMult3 vs EPD ring sums for both
UrQMD and FastOffline data at 14.5 GeV.
"""
#

import uproot as up
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import skew, kurtosis
import os

# Import and sort the distributions.
proton_params = pd.read_pickle(r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results\BigList\out_all.pkl")
UrQMD_params = up.open(r"D:\UrQMD_cent_sim\15\CentralityNtuple.root")
print(proton_params)
proton_params.sort_values(by=["refmult"], inplace=True)
# First we'll get everything by integer valued RefMult3.
RefCuts = [0, 10, 21, 41, 72, 118, 182, 270, 392, 472]  # Yu's determination of RefMult3 centrality via Glauber.
ref_data = proton_params['refmult'].to_numpy()
rings_data = []
for i in range(1, 33):
    rings_data.append(proton_params['ring{}'.format(i)].to_numpy())
rings_data = np.array(rings_data)
data = np.vstack((ref_data, rings_data))

ref_sim = ak.to_numpy(UrQMD_params['Rings']['RefMult3'].array())
rings_sim = []
for i in range(1, 17):
    rings_sim.append(ak.to_numpy(UrQMD_params['Rings']['r{0:0=2d}'.format(i)].array()))
rings_sim = np.array(rings_sim)
sim = np.vstack((ref_sim, rings_sim))

fig, ax = plt.subplots(4, 4, figsize=(16, 9), constrained_layout=True)
for i in range(4):
    for j in range(4):
        x = i*4 + j
        counter, binsX, binsY = np.histogram2d(sim[x+1], sim[0], bins=200)
        X, Y = np.meshgrid(binsY, binsX)
        graph = ax[i, j].pcolormesh(X, Y, counter, cmap="jet", norm=colors.LogNorm())
        ax[i, j].set_title("Ring{}".format(x+1), fontsize=20)
        fig.colorbar(graph, ax=ax[i, j])
plt.show()

fig, ax = plt.subplots(4, 4, figsize=(16, 9), constrained_layout=True)
for i in range(4):
    for j in range(4):
        x = i*4 + j
        counter, binsX, binsY = np.histogram2d(data[x+1] + data[x+17], data[0], bins=200)
        X, Y = np.meshgrid(binsY, binsX)
        graph = ax[i, j].pcolormesh(X, Y, counter, cmap="jet", norm=colors.LogNorm())
        ax[i, j].set_title("Ring{}".format(x+1), fontsize=20)
        fig.colorbar(graph, ax=ax[i, j])
plt.show()
