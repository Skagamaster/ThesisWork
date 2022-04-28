# \author Skipper Kagamaster
# \date 08/25/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
"""
The purpose of this code is to evaluate the cumulants of the net proton distributions from STAR
14.5 GeV FastOffline data generated using Yu Zhang's code on RCAF (from the STAR Collaboration).
This macro is mainly to find the cumulants from the distributions to vet the selection criteria.
"""

import uproot as up
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.stats import skew, kurtosis
import os

data = up.open(r'C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results\moments.root')['PtTree']
refmult = data['refMult3'].array(library='np')
pro = data['Np'].array(library='np')
apro = data['Nap'].array(library='np')
proton = pro - apro

uref = np.unique(refmult)
pro_cums = [[], [], [], []]
for i in uref:
    index = refmult == i
    arr = proton[index]
    arr = arr[arr > 0]
    pro_cums[0].append(np.mean(arr))
    pro_cums[1].append(np.var(arr))
    pro_cums[2].append(skew(arr) * np.power(np.sqrt(np.var(arr)), 3))
    pro_cums[3].append(kurtosis(arr) * np.power(np.var(arr), 2))

plt.hist2d(refmult[proton > 0], proton[proton > 0], bins=(700, 40), range=((0, 700), (0, 40)), cmin=1,
           norm=colors.LogNorm(), cmap='jet')
plt.colorbar()
plt.show()

plt.plot(uref, pro_cums[0])
plt.show()
