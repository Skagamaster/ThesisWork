# \author Skipper Kagamaster
# \date 06/23/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
"""
The purpose of this code is to evaluate the cumulants of the net proton distributions from STAR
14.5 GeV FastOffline data generated using the selection criteria from Yu Zhang of the STAR Collaboration.
This macro is mainly to find the cumulants fromt he distributions, but also it can serve as a
jumping off point for further analysis, such as building arrays for EPD centrality analysis.
"""
#

import uproot as up
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import os

# Import and sort the distributions.
# proton_params = pd.read_pickle(r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results\BigList\out_all.pkl")
# yu_params = up.open(r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Fast14.5_Result\Cumulant_NotCorrected.root")
# proton_params = proton_params[:int(1e5)]
proton_params = np.loadtxt(r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results\BigList\out_1.txt")[:, 1:5]
# proton_params = pd.DataFrame(proton_params, columns=["refmult", "protons", "antiprotons", "net_protons"])
yu_params = up.open(r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results\noCbwc.root")
proton_params.sort_values(by=["refmult"], inplace=True)
# First we'll get everything by integer valued RefMult3.
RefCuts = [0, 10, 21, 41, 72, 118, 182, 270, 392, 472]  # Yu's determination of RefMult3 centrality via Glauber.
refmult = proton_params['refmult'].to_numpy()
unique_refmult = np.unique(refmult)
size = len(unique_refmult)
net_protons = proton_params['net_protons'].to_numpy()
pro_cumulants_int = np.zeros((4, size))
for i in range(size):
    pro_int = net_protons[refmult == unique_refmult[i]]
    pro_cumulants_int[0][i] = np.mean(pro_int)
    pro_cumulants_int[1][i] = np.var(pro_int)
    pro_cumulants_int[2][i] = skew(pro_int) * np.power(np.sqrt(np.var(pro_int)), 3)
    pro_cumulants_int[3][i] = kurtosis(pro_int) * np.power(np.var(pro_int), 2)
# Let's similarly sort Yu's values.
yu_cumulants_int = [yu_params["NetC1"].values()[:int(np.max(unique_refmult))],
                    yu_params["NetC2"].values()[:int(np.max(unique_refmult))],
                    yu_params["NetC3"].values()[:int(np.max(unique_refmult))],
                    yu_params["NetC4"].values()[:int(np.max(unique_refmult))]]
yu_x = np.linspace(0, np.max(unique_refmult)-1, int(np.max(unique_refmult)))
print(len(yu_x), len(yu_cumulants_int[0]))

# Now let's see how we did. TODO Compare with Yu's values from his FastOffline runs.
fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
titles = [r"$\mu$", r"$\sigma^2$", r"$S\sigma$", r"$\kappa\sigma^2$"]
for i in range(2):
    for j in range(2):
        x = 2*i + j
        ax[i, j].scatter(unique_refmult, pro_cumulants_int[x], label="My Results", alpha=0.5)
        ax[i, j].scatter(yu_x, yu_cumulants_int[x], label="Yu's Results", marker="*", alpha=0.5)
        ax[i, j].set_xlabel("RefMult3", fontsize=15)
        ax[i, j].set_ylabel(titles[x], fontsize=20)
ax[1, 1].legend()
plt.suptitle("Cumulants Comparison", fontsize=30)
plt.show()
