# \author Skipper Kagamaster
# \date 10/28/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
#

"""
This code is made to run over UrQMD simulation data and
derive net proton cumulants using different means of
centrality measurements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Cumulants_functions as fn

# TODO Will need to add ML libraries once I have those correlation arrays.

# Import the pandas event and track arrays.
event = pd.read_parquet(r'F:\UrQMD\14\event.parquet.gzip')
print("Event df imported.")
track = pd.read_parquet(r'F:\UrQMD\14\track.parquet.gzip')
print("Track df imported.")

fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
ax[0, 0].hist(event['refmult'].to_numpy(), histtype='step', bins=300, density=True)
ax[0, 0].set_yscale('log')
ax[0, 0].set_xlabel(r"RefMult3", fontsize=20)
ax[0, 0].set_ylabel(r"$\frac{dN}{dRefMult3}$", fontsize=20)

ax[0, 1].hist(event['refmult_full'].to_numpy(), histtype='step', bins=300, density=True)
ax[0, 1].set_yscale('log')
ax[0, 1].set_xlabel(r"RefMult1", fontsize=20)
ax[0, 1].set_ylabel(r"$\frac{dN}{dRefMult1}$", fontsize=20)

ax[1, 0].hist(track['eta'].to_numpy(), histtype='step', bins=300, density=True)
# ax[1, 0].set_yscale('log')
ax[1, 0].set_xlabel(r"$\eta$", fontsize=20)
ax[1, 0].set_ylabel(r"$\frac{dN}{d\eta}$", fontsize=20)

ax[1, 1].hist(track['p_t'].to_numpy(), histtype='step', bins=300, density=True, range=(0, 6.0))
ax[1, 1].set_yscale('log')
ax[1, 1].set_xlabel(r"$p_T$", fontsize=20)
ax[1, 1].set_ylabel(r"$\frac{dN}{dp_T}$", fontsize=20)

# plt.show()
plt.close()

pos_percents = [20, 30, 40, 50, 60, 70, 80, 90, 95]
neg_percents = [80, 70, 60, 50, 40, 30, 20, 10, 5]
# pos_percents = np.linspace(1, 99, 99)
ref_quant = np.percentile(event['refmult'].to_numpy(), pos_percents)
ref_full_quant = np.percentile(event['refmult_full'].to_numpy(), pos_percents)
b_quant = np.percentile(event['b'].to_numpy(), neg_percents)

ref_cumu, ref_cumu_err, ref_cumu_vals = fn.proton_cumu_array_cbwc(proton_df=track,
                                                                  refmult_df=event,
                                                                  ref_quant=ref_quant,
                                                                  name='refmult')
ref_full_cumu, ref_full_cumu_err, ref_full_cumu_vals = fn.proton_cumu_array_cbwc(proton_df=track,
                                                                                 refmult_df=event,
                                                                                 ref_quant=ref_full_quant,
                                                                                 name='refmult_full')

fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
cum_lab = [r"$C_1$", r"$\frac{C_2}{C_1}$", r"$\frac{C_3}{C_2}$", r"$\frac{C_4}{C_2}$"]
x = ['80-100%', '70-80%', '60-70%', '50-60%', '40-50%',
     '30-40%', '20-30%', '10-20%', '5-10%', '0-5%']
print(len(x))
print(len(ref_cumu[0]))
print(ref_cumu_err[0])
np.save(r'F:\UrQMD\14\ref_cumu.npy', ref_cumu)
np.save(r'F:\UrQMD\14\ref_full_cumu.npy', ref_full_cumu)
np.save(r'F:\UrQMD\14\ref_cumu_err.npy', ref_cumu_err)
np.save(r'F:\UrQMD\14\ref_full_cumu_err.npy', ref_full_cumu_err)
np.save(r'F:\UrQMD\14\ref_cumu_vals.npy', ref_cumu_vals)
np.save(r'F:\UrQMD\14\ref_full_cumu_vals.npy', ref_full_cumu_vals)

for i in range(2):
    for j in range(2):
        x = i * 2 + j
        ax[i, j].errorbar((x, ref_cumu[x]), yerr=ref_cumu_err[x],
                          lw=0, marker="*", ms=10, alpha=0.5, label='refmult3',
                          elinewidth=1)
        ax[i, j].errorbar((x, ref_full_cumu[x]), yerr=ref_full_cumu_err[x],
                          lw=0, marker="^", ms=10, alpha=0.5, label='refmult',
                          elinewidth=1)
        ax[i, j].legend()
        ax[i, j].set_ylabel(cum_lab[x], fontsize=25)
plt.show()
