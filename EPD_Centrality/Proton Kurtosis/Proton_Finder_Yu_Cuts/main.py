# \author Skipper Kagamaster
# \date 03/20/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
#

import sys
import os
import typing
import logging
import numpy as np
import awkward as ak
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd

import pico_reader as pr
from scipy.stats import skew, kurtosis


time_start = time.perf_counter()

dataDirect = r"E:\2019Picos\14p5GeV\Runs"
saveDirect = r"C:\Users\dansk\Documents\Thesis\Protons\WIP\Protons"
finalDirect = r"C:\Users\dansk\Documents\Thesis\Protons\WIP\Protons\YuCuts\Histograms"

os.chdir(saveDirect)
bad_runs = np.loadtxt("badlist.txt").astype(int)

# Yu's cuts for RefMult3:
RefCuts = np.asarray((0, 10, 21, 41, 72, 118, 182, 270, 392, 472))

# This is to save our protons and refmult3, if doing the full analysis.
protons_full = []
antiprotons_full = []
refmult3_full = []

# The following is to hold the average values for an event.
# The dataframe holds both the average and the error. I will
# calculate these after basic vertex and track cuts.
averages = pd.DataFrame(columns=['ave_refmult3', 'err_refmults', 'ave_vz', 'err_vz', 'ave_vr', 'err_vr',
                                 'ave_pt', 'err_pt', 'ave_eta', 'err_eta', 'ave_zdcx', 'err_zdcx',
                                 'ave_phi', 'err_phi', 'ave_dca', 'err_dca'])

# Arrays to hold our histogram data for before and after QA cut analysis.
a, b, c, d = 1000, 161, 86, 101
vz_count, vz_bins = np.histogram(0, bins=a, range=(-200, 200))
vr_count, vr_binsX, vr_binsY = np.histogram2d([0], [0], bins=a, range=((-10, 10), (-10, 10)))
ref_count, ref_bins = np.histogram(0, bins=a, range=(0, a))
mpq_count, mpq_binsX, mpq_binsY = np.histogram2d([0], [0], bins=a, range=((0, 1.5), (-5, 5)))
rt_mult_count, rt_mult_binsX, rt_mult_binsY = np.histogram2d([0], [0], bins=(1700, a), range=((0, 1700), (0, 1000)))
rt_match_count, rt_match_binsX, rt_match_binsY = np.histogram2d([0], [0], bins=a, range=((0, 500), (0, 1000)))
ref_beta_count, ref_beta_binsX, ref_beta_binsY = np.histogram2d([0], [0], bins=(400, a), range=((0, 400), (0, 1000)))
pt_count, pt_bins = np.histogram(0, bins=a, range=(0, 6))
phi_count, phi_bins = np.histogram(0, bins=a, range=(-np.pi - 0.2, np.pi + 0.2))
dca_count, dca_bins = np.histogram(0, bins=a, range=(0, 5))
eta_count, eta_bins = np.histogram(0, bins=a, range=(-3, 3))
nhitsq_count, nhitsq_bins = np.histogram(0, bins=b, range=(-(b-1)/2, (b-1)/2))
nhits_dedx_count, nhits_dedx_bins = np.histogram(0, bins=c, range=(0, c-1))
betap_count, betap_binsX, betap_binsY = np.histogram2d([0], [0], bins=a, range=((0.5, 3.6), (0, 10)))
dedx_pq_count, dedx_pq_binsX, dedx_pq_binsY = np.histogram2d([0], [0], bins=a, range=((0, 31), (-3, 3)))

os.chdir(dataDirect)
r = len(os.listdir())
r_ = 1300  # For loop cutoff (to test on smaller batches).
count = 1

# Let's try using pandas; why not?
"""
df_events_init = pd.DataFrame(columns=["v_x", "v_y", "v_z", "v_r", "refmult3", "tofmult", "tofmatch", "beta_eta_1",
                                       "zdcx"])
df_events_cut = pd.DataFrame(columns=["v_x", "v_y", "v_z", "v_r", "refmult3", "tofmult", "tofmatch", "beta_eta_1",
                                      "zdcx"])
df_tracks_init = pd.DataFrame(columns=["p_t", "p_g", "phi", "dca", "eta", "nhitsfit", "nhitsdedx", "charge",
                                       "dedx", "rapidity", "nhitsmax", "nsigma_proton"])
df_tracks_ecut = pd.DataFrame(columns=["p_t", "p_g", "phi", "dca", "eta", "nhitsfit", "nhitsdedx", "charge",
                                       "dedx", "rapidity", "nhitsmax", "nsigma_proton"])
df_tracks_tcut = pd.DataFrame(columns=["p_t", "p_g", "phi", "dca", "eta", "nhitsfit", "nhitsdedx", "charge",
                                       "dedx", "rapidity", "nhitsmax", "nsigma_proton"])
df_toftracks_init = pd.DataFrame(columns=["p_t", "p_g", "phi", "dca", "eta", "nhitsfit", "nhitsdedx", "charge",
                                          "dedx", "rapidity", "nhitsmax", "nsigma_proton", "beta", "m_2"])
df_toftracks_ecut = pd.DataFrame(columns=["p_t", "p_g", "phi", "dca", "eta", "nhitsfit", "nhitsdedx", "charge",
                                          "dedx", "rapidity", "nhitsmax", "nsigma_proton", "beta", "m_2"])
df_toftracks_tcut = pd.DataFrame(columns=["p_t", "p_g", "phi", "dca", "eta", "nhitsfit", "nhitsdedx", "charge",
                                          "dedx", "rapidity", "nhitsmax", "nsigma_proton", "beta", "m_2"])
"""
# Now let's do some looping.
for file in sorted(os.listdir()):
    # This is to omit all runs marked "bad."
    run_num = file[:-5]
    # This cuts off the loop for testing.
    """
    if count < r_:
        continue
    
    r = 50
    if count > r:
        break
    """
    # This is just to show how far along the script is.
    if count % 5 == 0:
        print("Working on " + str(count) + " of " + str(r) + ".")
    # Runs to eliminate from consideration from QA.
    """
    # From Yu's list of bad runs.
    if int(run_num) in bad_runs:
        print("Run", run_num, "skipped for being marked bad.")
        print("Bad, I tell you. Bad!!")
        r -= 1
        continue
    # Yu's cutoff for nSigmaProton and dE/dx calibration issues.
    if int(run_num) > 20118040:
        print("Over the threshold Yu had set for display (20118040).")
        r -= 1
        break
    """
    # Import data from the pico.
    try:
        pico = pr.PicoDST()
        pico.import_data(file)
        v_z, v_y, v_x, refmult3, tofmult, tofmatch, beta_eta_1 = \
            pr.ak_to_numpy(pico.v_z, pico.v_y, pico.v_x, pico.refmult3, pico.tofmult, pico.tofmatch, pico.beta_eta_1)
        m_2, p_g, p_t, phi, dca, eta, nhitsfit, nhitsdedx, beta, charge, dedx = \
            pr.ak_to_numpy_flat(pico.m_2, pico.p_g, pico.p_t, pico.phi, pico.dca, pico.eta, pico.nhitsfit,
                                pico.nhitsdedx, pico.beta, pico.charge, pico.dedx)
        beta[beta == 0] = 1e-15  # To avoid infinities.
        p_g_tofpid, charge_tofpid = pr.index_cut(pico.tofpid, pico.p_g, pico.charge)
        p_g_tofpid, charge_tofpid = pr.ak_to_numpy_flat(p_g_tofpid, charge_tofpid)
        # Fill the histograms for pre QA cuts.
        vz_count += np.histogram(v_z, bins=a, range=(-200, 200))[0]
        vr_count += np.histogram2d(v_y, v_x, bins=a, range=((-10, 10), (-10, 10)))[0]
        ref_count += np.histogram(refmult3, bins=a, range=(0, a))[0]
        mpq_count += np.histogram2d(m_2, np.multiply(p_g_tofpid, charge_tofpid),
                                    bins=a, range=((0, 1.5), (-5, 5)))[0]
        rt_mult_count += np.histogram2d(tofmult, refmult3, bins=(1700, a), range=((0, 1700), (0, 1000)))[0]
        rt_match_count += np.histogram2d(tofmatch, refmult3, bins=a, range=((0, 500), (0, 1000)))[0]
        ref_beta_count += np.histogram2d(beta_eta_1, refmult3, bins=(400, a), range=((0, 400), (0, 1000)))[0]
        pt_count += np.histogram(p_t, bins=a, range=(0, 6))[0]
        phi_count += np.histogram(phi, bins=a, range=(-np.pi - 0.2, np.pi + 0.2))[0]
        dca_count += np.histogram(dca, bins=a, range=(0, 5))[0]
        eta_count += np.histogram(eta, bins=a, range=(-3, 3))[0]
        nhitsq_count += np.histogram(nhitsfit, bins=b, range=(-(b - 1) / 2, (b - 1) / 2))[0]
        nhits_dedx_count += np.histogram(nhitsdedx, bins=c, range=(0, c - 1))[0]
        betap_count += np.histogram2d(np.divide(1, beta), p_g_tofpid, bins=a, range=((0.5, 3.6), (0, 10)))[0]
        dedx_pq_count += np.histogram2d(dedx, np.multiply(charge, p_g), bins=a,
                                        range=((0, 31), (-3, 3)))[0]

        # This makes pandas dataframes of the events/tracks, but it's extremely memory intensive.
        """
        pico.make_df_events()
        frames = [df_events_init, pico.df]
        df_events_init = pd.concat(frames)
        df_events_init.reset_index(drop=True)
        pico.make_df_tracks()
        frames = [df_tracks_init, pico.df]
        df_tracks_init = pd.concat(frames)
        df_tracks_init.reset_index(drop=True)
        """
        pico.event_cuts_vertex()  # <- Primary vertex cuts
        pico.track_cuts_qa()  # <- Standard QA cuts on tracks

        # Fill our average values.
        df = pd.DataFrame([np.hstack((pr.get_ave(pico.refmult3), pr.get_ave(pico.v_z), pr.get_ave(pico.v_r),
                                      pr.get_ave(pico.p_t), pr.get_ave(pico.eta), pr.get_ave(pico.zdcx),
                                      pr.get_ave(pico.phi), pr.get_ave(pico.dca)))],
                          columns=['ave_refmult3', 'err_refmults', 'ave_vz', 'err_vz', 'ave_vr', 'err_vr', 'ave_pt',
                                   'err_pt', 'ave_eta', 'err_eta', 'ave_zdcx', 'err_zdcx', 'ave_phi', 'err_phi',
                                   'ave_dca', 'err_dca'])
        # frames = [averages, df]
        # averages = pd.concat(frames)
        # averages.reset_index()
    except Exception as e:  # For any issues that might pop up.
        print(e.__class__, "occurred in", run_num)
        count += 1
        continue
    count += 1

# Before and after plots.
"""
fig, ax = plt.subplots(2, figsize=(16, 9), constrained_layout=True)
titles = ["After QA Cuts", "After Proton Selection"]
for i in range(2):
    X, Y = np.meshgrid(bins[i][1], bins[1][0])
    imDp = ax[i].pcolormesh(X, Y, counts[i], cmap="jet", norm=colors.LogNorm())
    ax[i].set_xlabel(r"p*q $(\frac{GeV}{c})$", fontsize=10)
    ax[i].set_ylabel(r"$\frac{dE}{dX} (\frac{KeV}{cm})$", fontsize=10)
    ax[i].set_title(titles[i], fontsize=20)
    fig.colorbar(imDp, ax=ax[i])
plt.show()
plt.close()

protons = np.asarray(np.hstack(protons).flatten())
antiprotons = np.asarray(np.hstack(antiprotons).flatten())
refmult3 = np.asarray(np.hstack(refmult3).flatten())
print("Events: " + str(len(protons)))
cumulants_cbwc, cumulants_no_cbwc, cumulants_all, ref_set = pr.cbwc(protons, antiprotons, refmult3)
cumulant_names = (r"$\mu$", r"$\sigma^2$", "Skew", "Kurtosis")
RefCuts_labels = ["0-5%", "5-10%", "10-20%", "20-30%", "30-40%", "40-50%",
                  "50-60%", "60-70%", "70-80%", "80-100%"]
# Yu's cumulant values (CBWC, uncorrected for efficiencies).
C = [[0.5, 0.95, 1.58, 2.88, 4.61, 7.09, 10.59, 14.0, 16.78],
     [0.17, 0.78, 1.65, 2.99, 4.85, 7.53, 11.26, 14.89, 17.79],
     [0.17, 0.63, 1.30, 2.39, 3.87, 6.10, 9.10, 11.96, 13.79],
     [0.51, 1.10, 1.78, 2.54, 4.24, 6.53, 9.75, 13.05, 15.34]]

# Plot of my cumulants, cbwc cumulants, and Yu's cbwc cumulants.
fig, ax = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
for i in range(2):
    for j in range(2):
        x = i * 2 + j
        #ax[i, j].plot(RefCuts, cumulants_no_cbwc[x], c='b', marker="o", mfc='red', lw=0, mew=2,
        #              label="No CBWC", alpha=0.5)
        ax[i, j].plot(RefCuts, cumulants_cbwc[x], c='black', marker="o", mfc='red', lw=0, mew=2,
                      label="CBWC", alpha=0.5)
        ax[i, j].plot(RefCuts[1:], C[x], c='r', marker="o", mfc='green', lw=0, mew=2,
                      label="Yu's Values", alpha=0.5)
        ax[i, j].set_title(cumulant_names[x])
        #ax[i, j].set_xticks(RefCuts[::-1])
        #ax[i, j].set_xticklabels(RefCuts_labels, rotation=45)
        ax[i, j].set_xlabel("RefMult3")
        ax[i, j].grid(True)
        ax[i, j].legend()
plt.show()
plt.close()

# Raw cumulants plot by refmult3.
fig, ax = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
for i in range(2):
    for j in range(2):
        x = i * 2 + j
        ax[i, j].plot(ref_set, np.divide(cumulants_all[x+1], cumulants_all[0]), c='b', label="Raw Cumulants")
        ax[i, j].set_title(cumulant_names[x])
        ax[i, j].set_xlabel("RefMult3")
        ax[i, j].grid(True)
plt.suptitle("Raw Cumulants")
plt.show()
plt.close()

# Save the protons! Those poor protons.
os.chdir(finalDirect)
np.save("protons_yu.npy", protons)
np.save("antiprotons_yu.npy", antiprotons)
np.save("refmult3_yu.npy", refmult3)
"""
os.chdir(finalDirect)
averages.to_pickle("averages.pkl")
np.save("vz_hist.npy", (vz_count, vz_bins))
np.save("vr_hist.npy", (vr_count, vr_binsX, vr_binsY))
np.save("ref_hist.npy", (ref_count, ref_bins))
np.save("mpq_hist.npy", (mpq_count, mpq_binsX, mpq_binsY))
np.save("rt_mult_hist.npy", (rt_mult_count, rt_mult_binsX, rt_mult_binsY))
np.save("rt_match_hist.npy", (rt_match_count, rt_match_binsX, rt_match_binsY))
np.save("ref_beta_hist.npy", (ref_beta_count, ref_beta_binsX, ref_beta_binsY))
np.save("pt_hist.npy", (pt_count, pt_bins))
np.save("phi_hist.npy", (phi_count, phi_bins))
np.save("dca_hist.npy", (dca_count, dca_bins))
np.save("eta_hist.npy", (eta_count, eta_bins))
np.save("nhitsq_hist.npy", (nhitsq_count, nhitsq_bins))
np.save("nhits_dedx_hist.npy", (nhits_dedx_count, nhits_dedx_bins))
np.save("betap_hist.npy", (betap_count, betap_binsX, betap_binsY))
np.save("dedx_pq_hist.npy", (dedx_pq_count, dedx_pq_binsX, dedx_pq_binsY))
# For saving dataframes.
"""
df_events_init.to_pickle("df_events_init.pkl")
df_tracks_init.to_pickle("df_tracks_init.pkl")
"""
