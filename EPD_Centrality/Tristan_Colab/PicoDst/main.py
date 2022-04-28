# \author Skipper Kagamaster
# \date 03/20/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
#

# Not all are in use at present.
import sys
import os
import typing
import logging
import numpy as np
import awkward as ak
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pico_reader as pr
from scipy.stats import skew, kurtosis
import pandas as pd

# Directory where your picos live and where you want to save stuff.
dataDirect = r"E:\2019Picos\14p5GeV\Runs"
saveDirect = r"C:\Users\dansk\Documents\Thesis\Protons\WIP\QA_4_14_2021"
os.chdir(dataDirect)
r = len(os.listdir())
r_ = 1300  # For loop cutoff (to test on later picos).
count = 1

# Just a simple histogram for testing (using pg vs dedx).
a = 500
counts = np.zeros((2, a, a))
bins = np.zeros((2, 2, a+1))

protons = np.empty(0)
antiprotons = np.empty(0)
refmult3 = np.empty(0)

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
runs = np.empty(0)

# Loop over the picos.
for file in sorted(os.listdir()):
    run_num = file[:-5]
    # This takes out runs already gone over.

    if int(run_num) <= 20142059:
        print(run_num)
        continue

    # This cuts off the loop for testing.
    """
    if count < r_:
        continue
    """
    r = 5
    if count > r:
        break

    # This is just to show how far along the script is.
    if count % 5 == 0:
        print("Working on " + str(count) + " of " + str(r) + ".")
    
    # Import data from the pico.
    try:
        pico = pr.PicoDST()
        pico.import_data(file)
        runs = np.hstack((runs, pico.run_id))
        v_z, v_y, v_x, refmult3, tofmult, tofmatch, beta_eta_1 = \
            pr.ak_to_numpy(pico.v_z, pico.v_y, pico.v_x, pico.refmult3, pico.tofmult, pico.tofmatch, pico.beta_eta_1)
        m_2, p_g, p_t, phi, dca, eta, nhitsfit, nhitsdedx, beta, charge, dedx = \
            pr.ak_to_numpy_flat(pico.m_2, pico.p_g, pico.p_t, pico.phi, pico.dca, pico.eta, pico.nhitsfit,
                                pico.nhitsdedx, pico.beta, pico.charge, pico.dedx)
        beta[beta == 0] = 1e-15  # To avoid infinities.
        p_g_tofpid, charge_tofpid = pr.index_cut(pico.tofpid, pico.p_g, pico.charge)
        p_g_tofpid, charge_tofpid = pr.ak_to_numpy_flat(p_g_tofpid, charge_tofpid)
        # Fill the histograms for pre QA cuts.
        vz_count, vz_bins = np.histogram(v_z, bins=a, range=(-200, 200))
        vr_count, vr_binsX, vr_binsY = np.histogram2d(v_y, v_x, bins=a, range=((-10, 10), (-10, 10)))
        ref_count, ref_bins = np.histogram(refmult3, bins=a, range=(0, a))
        mpq_count, mpq_binsX, mpq_binsY = np.histogram2d(m_2, np.multiply(p_g_tofpid, charge_tofpid),
                                                         bins=a, range=((0, 1.5), (-5, 5)))
        rt_mult_count, rt_mult_binsX, rt_mult_binsY = np.histogram2d(tofmult, refmult3, bins=(1700, a),
                                                                     range=((0, 1700), (0, 1000)))
        rt_match_count, rt_match_binsX, rt_match_binsY = np.histogram2d(tofmatch, refmult3, bins=a,
                                                                        range=((0, 500), (0, 1000)))
        ref_beta_count, ref_beta_binsX, ref_beta_binsY = np.histogram2d(beta_eta_1, refmult3, bins=(400, a),
                                                                        range=((0, 400), (0, 1000)))
        pt_count, pt_bins = np.histogram(p_t, bins=a, range=(0, 6))
        phi_count, phi_bins = np.histogram(phi, bins=a, range=(-np.pi - 0.2, np.pi + 0.2))
        dca_count, dca_bins = np.histogram(dca, bins=a, range=(0, 5))
        eta_count, eta_bins = np.histogram(eta, bins=a, range=(-3, 3))
        nhitsq_count, nhitsq_bins = np.histogram(nhitsfit, bins=b, range=(-(b - 1) / 2, (b - 1) / 2))
        nhits_dedx_count, nhits_dedx_bins = np.histogram(nhitsdedx, bins=c, range=(0, c-1))
        betap_count, betap_binsX, betap_binsY = np.histogram2d(np.divide(1, beta), p_g_tofpid, bins=a,
                                                               range=((0.5, 3.6), (0, 10)))
        dedx_pq_count, dedx_pq_binsX, dedx_pq_binsY = np.histogram2d(dedx, np.multiply(charge, p_g), bins=a,
                                                                     range=((0, 31), (-3, 3)))
        # Fill our average values.
        df = pd.DataFrame([np.hstack((pr.get_ave(pico.refmult3), pr.get_ave(pico.v_z), pr.get_ave(pico.v_r),
                                      pr.get_ave(pico.p_t), pr.get_ave(pico.eta), pr.get_ave(pico.zdcx),
                                      pr.get_ave(pico.phi), pr.get_ave(pico.dca)))],
                          columns=['ave_refmult3', 'err_refmults', 'ave_vz', 'err_vz', 'ave_vr', 'err_vr', 'ave_pt',
                                   'err_pt', 'ave_eta', 'err_eta', 'ave_zdcx', 'err_zdcx', 'ave_phi', 'err_phi',
                                   'ave_dca', 'err_dca'])
        frames = [averages, df]
        averages = pd.concat(frames)
        averages.reset_index()
        file = r"C:\Users\dansk\Documents\Thesis\Protons\WIP\QA_4_14_2021\{}".format(pico.run_id)
        df.to_pickle(file + "ave.pkl")
        np.save(file+"vz_hist.npy", (vz_count, vz_bins))
        np.save(file+"vr_hist.npy", (vr_count, vr_binsX, vr_binsY))
        np.save(file+"ref_hist.npy", (ref_count, ref_bins))
        np.save(file+"mpq_hist.npy", (mpq_count, mpq_binsX, mpq_binsY))
        np.save(file+"rt_mult_hist.npy", (rt_mult_count, rt_mult_binsX, rt_mult_binsY))
        np.save(file+"rt_match_hist.npy", (rt_match_count, rt_match_binsX, rt_match_binsY))
        np.save(file+"ref_beta_hist.npy", (ref_beta_count, ref_beta_binsX, ref_beta_binsY))
        np.save(file+"pt_hist.npy", (pt_count, pt_bins))
        np.save(file+"phi_hist.npy", (phi_count, phi_bins))
        np.save(file+"dca_hist.npy", (dca_count, dca_bins))
        np.save(file+"eta_hist.npy", (eta_count, eta_bins))
        np.save(file+"nhitsq_hist.npy", (nhitsq_count, nhitsq_bins))
        np.save(file+"nhits_dedx_hist.npy", (nhits_dedx_count, nhits_dedx_bins))
        np.save(file+"betap_hist.npy", (betap_count, betap_binsX, betap_binsY))
        np.save(file+"dedx_pq_hist.npy", (dedx_pq_count, dedx_pq_binsX, dedx_pq_binsY))
        np.save(file+"runs.npy", runs)
        """
        # This is for finding protons.
        pico_low = pr.PicoDST()
        pico_high = pr.PicoDST()
        pico_low.import_data(file)
        pico_high.import_data(file)
        # Pre event cut histogram of pq vs dedx
        counts1, binsX1, binsY1 = np.histogram2d(ak.to_numpy(ak.flatten(pico_low.dedx)),
                                                 ak.to_numpy(ak.flatten(pico_low.p_g)) *
                                                 ak.to_numpy(ak.flatten(pico_low.charge)),
                                                 bins=a, range=((0, 20), (-5, 5)))
        # Event cuts and the same histogram after.
        pico_low.vertex_cuts()
        pico_low.refmult_correlation_cuts()
        pico_low.calibrate_nsigmaproton()
        pico_high.vertex_cuts()
        pico_high.refmult_correlation_cuts()
        pico_high.calibrate_nsigmaproton()
        pico_low.track_qa_cuts()
        pico_low.select_protons_low()
        pico_high.track_qa_cuts_tof()
        pico_high.select_protons_high()
        dedx = np.concatenate((ak.to_numpy(ak.flatten(pico_low.dedx)), ak.to_numpy(ak.flatten(pico_high.dedx))))
        p_g = np.concatenate((ak.to_numpy(ak.flatten(pico_low.p_g)), ak.to_numpy(ak.flatten(pico_high.p_g))))
        charge = np.concatenate((ak.to_numpy(ak.flatten(pico_low.charge)), ak.to_numpy(ak.flatten(pico_high.charge))))
        counts2, binsX2, binsY2 = np.histogram2d(dedx, p_g * charge,
                                                 bins=a, range=((0, 20), (-5, 5)))
        counts += (counts1, counts2)
        bins = ((binsX1, binsY1), (binsX2, binsY2))
        pro = np.concatenate((ak.to_numpy(pico_low.protons), ak.to_numpy(pico_high.protons)))
        antipro = np.concatenate((ak.to_numpy(pico_low.antiprotons), ak.to_numpy(pico_high.antiprotons)))
        refs = np.concatenate((ak.to_numpy(pico_low.refmult3), ak.to_numpy(pico_high.refmult3)))
        protons = np.concatenate((protons, pro))
        antiprotons = np.concatenate((antiprotons, antipro))
        refmult3 = np.concatenate((refmult3, refs))
        """
    except Exception as e:  # For any issues that might pop up.
        print(e.__class__, "occurred in", run_num)
        count += 1
        continue
    count += 1

"""
fig, ax = plt.subplots(2, figsize=(16, 9), constrained_layout=True)
titles = ["Before Event Cuts", "After Event Cuts"]
for i in range(2):
    X, Y = np.meshgrid(bins[i][1], bins[1][0])
    imDp = ax[i].pcolormesh(X, Y, counts[i], cmap="jet", norm=colors.LogNorm())
    ax[i].set_xlabel(r"p*q $(\frac{GeV}{c})$", fontsize=10)
    ax[i].set_ylabel(r"$\frac{dE}{dX} (\frac{KeV}{cm})$", fontsize=10)
    ax[i].set_title(titles[i], fontsize=20)
    fig.colorbar(imDp, ax=ax[i])
plt.show()
plt.close()

os.chdir(saveDirect)
np.save("protons.npy", protons)
np.save("antiprotons.npy", antiprotons)
np.save("refmult3.npy", refmult3)
"""
