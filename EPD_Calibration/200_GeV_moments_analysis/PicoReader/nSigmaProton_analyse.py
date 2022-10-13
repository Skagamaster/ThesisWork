#
# \Just a little code to check out nSigmaProton calibrations.
#
# \author Skipper Kagamaster
# \date 05/06/2022
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
# /

import numpy as np
import matplotlib.pyplot as plt
import os

nsigs = np.load(r'D:\14GeV\Thesis\PythonArrays\nSigmaCal.npy', allow_pickle=True).T
nsig_ = []
for i in nsigs:
    nsig_.append(np.hstack(i))
nsig_ = np.asarray(nsig_)
runs = np.load(r'D:\14GeV\Thesis\PythonArrays\runs.npy', allow_pickle=True)

pt_vals = [r'$p_T$ = 0.1-0.2 $\frac{GeV}{c}$', r'$p_T$ = 0.2-0.3 $\frac{GeV}{c}$',
           r'$p_T$ = 0.3-0.4 $\frac{GeV}{c}$', r'$p_T$ = 0.4-0.5 $\frac{GeV}{c}$',
           r'$p_T$ = 0.5-0.6 $\frac{GeV}{c}$', r'$p_T$ = 0.6-0.7 $\frac{GeV}{c}$',
           r'$p_T$ = 0.7-0.8 $\frac{GeV}{c}$', r'$p_T$ = 0.8-0.9 $\frac{GeV}{c}$',
           r'$p_T$ = 0.9-1.0 $\frac{GeV}{c}$', r'$p_T$ = 1.0-1.1 $\frac{GeV}{c}$',
           r'$p_T$ = 1.1-1.2 $\frac{GeV}{c}$']

fig, ax = plt.subplots(3, 4, figsize=(16, 9), constrained_layout=True)
for i in range(3):
    for j in range(4):
        n = i*4 + j
        if n == 11:
            ax[i, j].set_axis_off()
            continue
        ax[i, j].hist(nsig_[n], bins=30, histtype='step', density=True, color='purple', lw=1)
        ax[i, j].set_xlabel(r'$n\sigma_p$', fontsize=15, loc='right')
        ax[i, j].set_ylabel(r'$\frac{dN}{dn\sigma_p}$', fontsize=15)
        ax[i, j].set_title(pt_vals[n], fontsize=15)
        ax[i, j].set_yscale('log')
plt.show()
