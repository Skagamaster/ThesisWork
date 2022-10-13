# \Display QA quantities from FastOffline data (with my new cuts)
#
#
# \author Skipper Kagamaster
# \date 06/01/2022
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

os.chdir(r"D:\14GeV\Thesis\PythonArrays\Analysis_Histograms")
#################
# 1D histograms #
#################
v_z_counts = np.load('v_z.npy', allow_pickle=True)
v_z_bins = np.load('v_z_bins.npy', allow_pickle=True)
p_t_count = np.load('p_t.npy', allow_pickle=True)
p_t_bins = np.load('p_t_bins.npy', allow_pickle=True)
phi_count = np.load('phi.npy', allow_pickle=True)
phi_bins = np.load('phi_bins.npy', allow_pickle=True)
dca_count = np.load('dca.npy', allow_pickle=True)
dca_bins = np.load('dca_bins.npy', allow_pickle=True)
eta_count = np.load('eta.npy', allow_pickle=True)
eta_bins = np.load('eta_bins.npy', allow_pickle=True)
rap_count = np.load('rap.npy', allow_pickle=True)
rap_bins = np.load('rap_bins.npy', allow_pickle=True)
nhq_count = np.load('nhq.npy', allow_pickle=True)
nhq_bins = np.load('nhq_bins.npy', allow_pickle=True)
nhde_count = np.load('nhde.npy', allow_pickle=True)
nhde_bins = np.load('nhde_bins.npy', allow_pickle=True)
nhr_count = np.load('nhr.npy', allow_pickle=True)
nhr_bins = np.load('nhr_bins.npy', allow_pickle=True)
pre_cut_counts = [v_z_counts, p_t_count, phi_count, dca_count,
                  eta_count, rap_count]
pre_cut_bins = [v_z_bins, p_t_bins, phi_bins, dca_bins,
                eta_bins, rap_bins]
# After cuts
av_z_counts = np.load('av_z.npy', allow_pickle=True)
ap_t_count = np.load('ap_t.npy', allow_pickle=True)
aphi_count = np.load('aphi.npy', allow_pickle=True)
adca_count = np.load('adca.npy', allow_pickle=True)
aeta_count = np.load('aeta.npy', allow_pickle=True)
arap_count = np.load('arap.npy', allow_pickle=True)
anhq_count = np.load('anhq.npy', allow_pickle=True)
anhde_count = np.load('anhde.npy', allow_pickle=True)
anhr_count = np.load('anhr.npy', allow_pickle=True)
aft_cut_counts = [av_z_counts, ap_t_count, aphi_count, adca_count,
                  aeta_count, arap_count]
#################
# 2D histograms #
#################
v_r_counts = np.load('v_r.npy', allow_pickle=True)
v_r_binsX = np.load('v_r_binsX.npy', allow_pickle=True)
v_r_binsY = np.load('v_r_binsY.npy', allow_pickle=True)
rt_mult_counts = np.load('rt_mult.npy', allow_pickle=True)
rt_mult_binsX = np.load('rt_mult_binsX.npy', allow_pickle=True)
rt_mult_binsY = np.load('rt_mult_binsY.npy', allow_pickle=True)
rt_match_count = np.load('rt_match.npy', allow_pickle=True)
rt_match_binsX = np.load('rt_match_binsX.npy', allow_pickle=True)
rt_match_binsY = np.load('rt_match_binsY.npy', allow_pickle=True)
rb_count = rb_count = np.load('rb.npy', allow_pickle=True)
rb_binsX = np.load('rb_binsX.npy', allow_pickle=True)
rb_binsY = np.load('rb_binsY.npy', allow_pickle=True)
mpq_count = np.load('mpq.npy', allow_pickle=True)
mpq_binsX = np.load('mpq_binsX.npy', allow_pickle=True)
mpq_binsY = np.load('mpq_binsY.npy', allow_pickle=True)
bp_count = np.load('bp.npy', allow_pickle=True)
bp_binsX = np.load('bp_binsX.npy', allow_pickle=True)
bp_binsY = np.load('bp_binsY.npy', allow_pickle=True)
dEp_count = np.load('dEp.npy', allow_pickle=True)
dEp_binsX = np.load('dEp_binsX.npy', allow_pickle=True)
dEp_binsY = np.load('dEp_binsY.npy', allow_pickle=True)
pre_cut_counts_2d = [v_r_counts, rt_mult_counts, rb_count, dEp_count]
pre_cut_binsX = [v_r_binsX, rt_mult_binsX, rb_binsX, dEp_binsX]
pre_cut_binsY = [v_r_binsY, rt_mult_binsY, rb_binsY, dEp_binsY]
axes_2D = [[r"$v_x$ (cm)", r"$v_y$ (cm)"],
           [r"$X_{RM3}$", "TofMult"],
           [r"$X_{RM3}$", r"$\beta\eta1$"],
           [r"$p_G \cdot q$ (GeV/c)", r"$\frac{dE}{dx}$ (keV/cm)"]]
# After cuts
av_r_counts = np.load('av_r.npy', allow_pickle=True)
art_mult_counts = np.load('art_mult.npy', allow_pickle=True)
art_match_count = np.load('art_match.npy', allow_pickle=True)
arb_count = np.load('arb.npy', allow_pickle=True)
ampq_count = np.load('ampq.npy', allow_pickle=True)
abp_count = np.load('abp.npy', allow_pickle=True)
adEp_count = np.load('adEp.npy', allow_pickle=True)
aft_cut_counts_2d = [av_r_counts, art_mult_counts, arb_count, adEp_count]
# After PID
pmpq_count = np.load('pmpq.npy', allow_pickle=True)
pbp_count = np.load('pbp.npy', allow_pickle=True)
pdEp_count = np.load('pdEp.npy', allow_pickle=True)

# Now to do the 1D plots.
x_1D = [r"$v_z$ (cm)", r"$p_T$ (GeV/c)", r"$\phi$ (rad)", "DCA (cm)",
        r"$\eta$", "y"]
y_1D = [r"dN/d$v_z$", r"dN/d$p_T$", r"dN/d$\phi$", "dN/dDCA",
        r"dN/d$\eta$", "dN/dy"]

fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
for i in range(2):
    for j in range(3):
        k = i*3 + j
        ax[i, j].plot(pre_cut_bins[k][:-1], pre_cut_counts[k],
                      c='k', label="Raw")
        ax[i, j].plot(pre_cut_bins[k][:-1], aft_cut_counts[k],
                      c='r', label="Cut")
        ax[i, j].set_yscale('log')
        ax[i, j].set_xlabel(x_1D[k], loc='right', fontsize=15)
        ax[i, j].set_ylabel("Counts", fontsize=15)
        ax[i, j].legend()
# plt.show()
plt.close()

# And now the 2D plots.
fig, ax = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
for i in range(2):
    for j in range(2):
        k = i*2 + j
        X, Y = np.meshgrid(pre_cut_binsX[k], pre_cut_binsY[k])
        am = ax[i, j].pcolormesh(X, Y, pre_cut_counts_2d[k].T, cmap='bone', norm=LogNorm())
        im = ax[i, j].pcolormesh(X, Y, aft_cut_counts_2d[k].T, cmap='jet', norm=LogNorm())
        ax[i, j].set_xlabel(axes_2D[k][0], fontsize=12)
        ax[i, j].set_ylabel(axes_2D[k][1], fontsize=12)
        fig.colorbar(am, ax=ax[i, j])
        fig.colorbar(im, ax=ax[i, j])
        if k == 3:
            um = ax[i, j].pcolormesh(X, Y, pdEp_count.T, cmap='gnuplot', norm=LogNorm())
            fig.colorbar(um, ax=ax[i, j])
plt.show()
plt.close()
