# \author Skipper Kagamaster
# \date 09/14/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
#
"""
This code is used to generate plots for the STAR Collaboration meeting,
9/2021. This is to show all of the QA cuts that were done for the net
proton cumulant analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import uproot as up
import awkward as ak
import functions as fn

os.chdir(r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results\BigList\Runs")
file = "out_all.root"
pico = fn.HistoData(data_file=file)

# PID Plot for dE/dx
plt.figure(figsize=(8, 8))
plt.pcolormesh(pico.dedxX, pico.dedxY, pico.dedx_pq.T, cmap='bone', norm=LogNorm(), shading='auto')
plt.pcolormesh(pico.dedxX, pico.dedxY, pico.pdedx_pq.T, cmap='jet', norm=LogNorm(), shading='auto')
plt.xlabel(r"$p*q$ ($\frac{GeV}{c}$)", fontsize=20)
plt.ylabel(r"$\frac{dE}{dx}$ ($\frac{KeV}{cm}$)", fontsize=20)
plt.colorbar()
plt.tight_layout()
plt.show()

# 2D QA Plots
fig, ax = plt.subplots(2, 4, figsize=(16, 9), constrained_layout=True)
ax[0, 0].pcolormesh(pico.v_r_X, pico.v_r_Y, pico.v_r, cmap='bone', norm=LogNorm(), shading='auto')
im = ax[0, 0].pcolormesh(pico.v_r_X, pico.v_r_Y, pico.av_r, cmap='jet', norm=LogNorm(), shading='auto')
ax[0, 0].set_xlabel(r"$v_x$ (cm)", fontsize=15)
ax[0, 0].set_ylabel(r"$v_y$ (cm)", fontsize=15)
fig.colorbar(im, ax=ax[0, 0])
ax[0, 1].pcolormesh(pico.beta_p_X, pico.beta_p_Y, pico.betap.T, cmap='bone', norm=LogNorm(), shading='auto')
im = ax[0, 1].pcolormesh(pico.beta_p_X, pico.beta_p_Y, pico.abetap.T, cmap='jet', norm=LogNorm(), shading='auto')
ax[0, 1].set_xlabel(r"p ($\frac{GeV}{c}$)", fontsize=15)
ax[0, 1].set_ylabel(r"$\frac{1}{\beta}$", fontsize=15)
fig.colorbar(im, ax=ax[0, 1])
ax[0, 2].pcolormesh(pico.rtX, pico.rtY, pico.rt_mult.T, cmap='bone', norm=LogNorm(), shading='auto')
im = ax[0, 2].pcolormesh(pico.rtX, pico.rtY, pico.art_mult.T, cmap='jet', norm=LogNorm(), shading='auto')
ax[0, 2].set_xlabel(r"RefMult3", fontsize=15)
ax[0, 2].set_ylabel(r"TofMult", fontsize=15)
fig.colorbar(im, ax=ax[0, 2])
ax[0, 3].pcolormesh(pico.rtmX, pico.rtmY, pico.rt_match, cmap='bone', norm=LogNorm(), shading='auto')
im = ax[0, 3].pcolormesh(pico.rtmX, pico.rtmY, pico.art_match, cmap='jet', norm=LogNorm(), shading='auto')
ax[0, 3].set_xlabel(r"RefMult3", fontsize=15)
ax[0, 3].set_ylabel(r"TofMatch", fontsize=15)
fig.colorbar(im, ax=ax[0, 3])
ax[1, 0].pcolormesh(pico.r_bX, pico.r_bY, pico.ref_beta.T, cmap='bone', norm=LogNorm(), shading='auto')
im = ax[1, 0].pcolormesh(pico.r_bX, pico.r_bY, pico.aref_beta.T, cmap='jet', norm=LogNorm(), shading='auto')
ax[1, 0].set_xlabel(r"RefMult3", fontsize=15)
ax[1, 0].set_ylabel(r"$\beta\eta1$", fontsize=15)
fig.colorbar(im, ax=ax[1, 0])
ax[1, 1].pcolormesh(pico.mpX, pico.mpY, pico.mpq.T, cmap='bone', norm=LogNorm(), shading='auto')
im = ax[1, 1].pcolormesh(pico.mpX, pico.mpY, pico.ampq.T, cmap='jet', norm=LogNorm(), shading='auto')
ax[1, 1].set_xlabel(r"$p*q$ ($\frac{GeV}{c}$)", fontsize=15)
ax[1, 1].set_ylabel(r"$m^2$ ($\frac{GeV^2}{c^4}$)", fontsize=15)
fig.colorbar(im, ax=ax[1, 1])
ax[1, 2].pcolormesh(pico.dedxX, pico.dedxY, pico.dedx_pq.T, cmap='bone', norm=LogNorm(), shading='auto')
im = ax[1, 2].pcolormesh(pico.dedxX, pico.dedxY, pico.adedx_pq.T, cmap='jet', norm=LogNorm(), shading='auto')
ax[1, 2].set_xlabel(r"$p*q$ ($\frac{GeV}{c}$)", fontsize=15)
ax[1, 2].set_ylabel(r"$\frac{dE}{dx}$ ($\frac{KeV}{cm}$)", fontsize=15)
fig.colorbar(im, ax=ax[1, 2])
ax[1, 3].set_axis_off()
plt.show()
