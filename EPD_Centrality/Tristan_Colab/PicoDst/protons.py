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


def events_test(events):
    print(events.v_r)
    pass


# Function for event cuts.
def event_cuts(events):
    select = events.v_r <= 2.0  # transverse vertex within 2cm
    select = select & np.abs(events.v_z) <= 30.0  # and a z cut at +- 30.0 cm
    # Reject pileup
    select = select & (events.tofmult >= (1.22 * events.refmult3 - 24.29))
    select = select & events.tofmult <= (2.493 * events.refmult3 + 77.02)
    select = select & (events.tofmatch >= (0.379 * events.refmult3 - 8.6))
    select = select & events.tofmatch <= (0.6692 * events.refmult3 + 18.66)
    select = select & (events.beta_eta_1 >= (0.3268 * events.refmult3 - 11.07))
    return select


# Function for low pt track cuts.
def track_cuts_base(events):
    select = events.nhitsdedx > 5
    select = select & events.nhitsfit > 20
    select = select & np.divide(np.absolute(events.nhitsfit), events.nhitsmax) > 0.52
    select = select & events.dca <= 1.0
    select = select & events.p_t >= 0.2
    select = select & events.p_t <= 10.0
    select = select & np.abs(events.rapidity) <= 0.5
    select = select & np.abs(events.nsigma_proton) <= 2000.0
    return select


# Function for high pt track cuts.
def track_cuts_low(events):
    select = events.p_t > 0.4
    select = select & events.p_t < 0.8
    select = select & events.p_g <= 1.0
    return select


def track_cuts_high(events):  # MUST BE TOF MATCHED!
    select = events.p_t >= 0.8
    select = select & events.p_t < 2.0
    select = select & events.p_g <= 3.0
    select = select & events.m_2 >= 0.6
    select = select & events.m_2 <= 1.2
    return select


# Directory where your picos live and where you want to save stuff.
dataDirect = r"E:\2019Picos\14p5GeV\Runs"
saveDirect = r"C:\Users\dansk\Documents\Thesis\Protons\WIP\QA_4_14_2021\protons"

# Just a simple histogram for testing (using pg vs dedx).
a = 500
counts = np.zeros((2, a, a))
bins = np.zeros((2, 2, a+1))

# This is to hold all of the event data we want for proton and centrality analysis.
protons_df = pd.DataFrame(columns=['ring1', 'ring2', 'ring3', 'ring4', 'ring5', 'ring6', 'ring7', 'ring8',
                                   'ring9', 'ring10', 'ring11', 'ring12', 'ring13', 'ring14', 'ring15', 'ring16',
                                   'ring17', 'ring18', 'ring19', 'ring20', 'ring21', 'ring22', 'ring23', 'ring24',
                                   'ring25', 'ring26', 'ring27', 'ring28', 'ring29', 'ring30', 'ring31', 'ring32',
                                   'vzvpd', 'refmult3', 'protons', 'antiprotons', 'net_protons'])

# Arrays to hold our histogram data for after QA cut analysis.
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
os.chdir(dataDirect)
r = len(os.listdir())
r_ = 1300  # For loop cutoff (to test on later picos).
count = 1
event_total = 0
proton_found = np.empty(0)
antiproton_found = np.empty(0)
p_ave_low = np.empty(0)
p_ave_high = np.empty(0)
ap_ave_low = np.empty(0)
ap_ave_high = np.empty(0)
for file in sorted(os.listdir()):
    run_num = file[:-5]
    # This takes out runs already gone over.

    if int(run_num) <= 20113027:
        print(run_num)
        continue
    """
    # This cuts off the loop for testing.
    
    if count < r_:
        continue
    """
    # This is the batch size.
    r = 200
    if count > r:
        break

    # To omit any runs marked bad from QA average analysis.
    badruns = np.load(r"C:\Users\dansk\Documents\Thesis\Protons\WIP\QA_4_14_2021\protons\total\badruns.npy",
                      allow_pickle=True)
    if int(run_num) in badruns:
        print("Run", run_num, "was no good. Let's move on. Just ... let it go.")
        continue
    # This is just to show how far along the script is.
    if count % 5 == 0:
        print("Working on " + str(count) + " of " + str(r) + ".")
    
    # Import data from the pico.
    try:
        pico = pr.PicoDST()
        pico.import_data(file)
        runs = np.hstack((runs, pico.run_id)).astype('int')
        runs = runs[runs != np.array(None)]  # Eliminate runs with errors.
        pico.calibrate_nsigmaproton()  # To zero our nSigmaProton distributions by p_t bin.
        # Make the event cuts.
        pico.event_cuts()

        # For high p_t; only matters in track cuts.
        pico_h = pr.PicoDST()
        pico_h.import_data(file)
        pico_h.calibrate_nsigmaproton()  # To zero our nSigmaProton distributions by p_t bin.
        pico_h.event_cuts()
        event_total += len(pico.v_x)

        # Event quantities insensitive to track cuts for our df.
        ring_sum = pico.epd_hits.generate_epd_hit_matrix()
        vz_vpd = pico.vz_vpd
        refs = pico.refmult3

        # Histograms for after event cuts.
        v_z, v_y, v_x, refmult3, tofmult, tofmatch, beta_eta_1 = \
            pr.ak_to_numpy(pico.v_z, pico.v_y, pico.v_x, pico.refmult3, pico.tofmult, pico.tofmatch, pico.beta_eta_1)
        m_2, p_g, p_t, phi, dca, eta, nhitsfit, nhitsdedx, beta, charge, dedx = \
            pr.ak_to_numpy_flat(pico.m_2, pico.p_g, pico.p_t, pico.phi, pico.dca, pico.eta, pico.nhitsfit,
                                pico.nhitsdedx, pico.beta, pico.charge, pico.dedx)
        beta[beta == 0] = 1e-15  # To avoid infinities.
        p_g_tofpid, charge_tofpid = pr.index_cut(pico.tofpid, pico.p_g, pico.charge)
        p_g_tofpid, charge_tofpid = pr.ak_to_numpy_flat(p_g_tofpid, charge_tofpid)
        vz_count += np.histogram(v_z, bins=a, range=(-200, 200))[0]
        vr_count += np.histogram2d(v_y, v_x, bins=a, range=((-10, 10), (-10, 10)))[0]
        ref_count += np.histogram(refmult3, bins=a, range=(0, a))[0]
        mpq_count += np.histogram2d(m_2, np.multiply(p_g_tofpid, charge_tofpid), bins=a, range=((0, 1.5), (-5, 5)))[0]
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
        dedx_pq_count += np.histogram2d(dedx, np.multiply(charge, p_g), bins=a, range=((0, 31), (-3, 3)))[0]

        file = r"C:\Users\dansk\Documents\Thesis\Protons\WIP\QA_4_14_2021\protons\\"

        np.save(file + "vz_hist.npy", (vz_count, vz_bins))
        np.save(file + "vr_hist.npy", (vr_count, vr_binsX, vr_binsY))
        np.save(file + "ref_hist.npy", (ref_count, ref_bins))
        np.save(file + "mpq_hist.npy", (mpq_count, mpq_binsX, mpq_binsY))
        np.save(file + "rt_mult_hist.npy", (rt_mult_count, rt_mult_binsX, rt_mult_binsY))
        np.save(file + "rt_match_hist.npy", (rt_match_count, rt_match_binsX, rt_match_binsY))
        np.save(file + "ref_beta_hist.npy", (ref_beta_count, ref_beta_binsX, ref_beta_binsY))
        np.save(file + "pt_hist.npy", (pt_count, pt_bins))
        np.save(file + "phi_hist.npy", (phi_count, phi_bins))
        np.save(file + "dca_hist.npy", (dca_count, dca_bins))
        np.save(file + "eta_hist.npy", (eta_count, eta_bins))
        np.save(file + "nhitsq_hist.npy", (nhitsq_count, nhitsq_bins))
        np.save(file + "nhits_dedx_hist.npy", (nhits_dedx_count, nhits_dedx_bins))
        np.save(file + "betap_hist.npy", (betap_count, betap_binsX, betap_binsY))
        np.save(file + "dedx_pq_hist.npy", (dedx_pq_count, dedx_pq_binsX, dedx_pq_binsY))

        np.save(file + "runs.npy", runs)

        # Track cuts (high and low).
        pico.track_qa_cuts()
        pico_h.track_qa_cuts_tof()
        pico.select_protons_low()
        pico_h.select_protons_high()

        # Just a simple histogram to see how we're doing on selection.
        dedx_l = ak.to_numpy(ak.flatten(pico.dedx))
        dedx_h = ak.to_numpy(ak.flatten(pico_h.dedx))
        dedx = np.hstack((dedx_l, dedx_h))
        charge_l = ak.to_numpy(ak.flatten(pico.charge))
        charge_h = ak.to_numpy(ak.flatten(pico_h.charge))
        charge = np.hstack((charge_l, charge_h))
        p_g_l = ak.to_numpy(ak.flatten(pico.p_g))
        p_g_h = ak.to_numpy(ak.flatten(pico_h.p_g))
        p_g = np.hstack((p_g_l, p_g_h))
        dedx_pq_count, dedx_pq_binsX, dedx_pq_binsY = np.histogram2d(dedx, np.multiply(charge, p_g), bins=a,
                                                                     range=((0, 31), (-3, 3)))
        np.save(file + "protons_dedxpq_hist.npy", (dedx_pq_count, dedx_pq_binsX, dedx_pq_binsY))

        # PID quantities from track cuts for our df.
        p_low = ak.sum(ak.where(pico.charge > 0, 1, 0), axis=-1)
        ap_low = ak.sum(ak.where(pico.charge < 0, 1, 0), axis=-1)
        p_high = ak.sum(ak.where(pico_h.charge > 0, 1, 0), axis=-1)
        ap_high = ak.sum(ak.where(pico_h.charge < 0, 1, 0), axis=-1)
        p_ave_low = np.hstack((p_ave_low, np.mean(p_low)))
        ap_ave_low = np.hstack((ap_ave_low, np.mean(ap_low)))
        p_ave_high = np.hstack((p_ave_high, np.mean(p_high)))
        ap_ave_high = np.hstack((ap_ave_high, np.mean(ap_high)))
        np.save(file + 'low_proton_ave.npy', p_ave_low)
        np.save(file + 'low_antiproton_ave.npy', ap_ave_low)
        np.save(file + 'high_proton_ave.npy', p_ave_high)
        np.save(file + 'high_antiproton_ave.npy', ap_ave_high)

        protons = ak.to_numpy(p_low + p_high)
        antiprotons = ak.to_numpy(ap_low + ap_high)
        protons_net = protons - antiprotons
        proton_found = np.hstack((proton_found, np.mean(protons)))
        antiproton_found = np.hstack((antiproton_found, np.mean(antiprotons)))

        # Fill our proton event dataframe.
        df = pd.DataFrame(ring_sum.T, columns=['ring1', 'ring2', 'ring3', 'ring4', 'ring5', 'ring6', 'ring7', 'ring8',
                                               'ring9', 'ring10', 'ring11', 'ring12', 'ring13', 'ring14', 'ring15',
                                               'ring16', 'ring17', 'ring18', 'ring19', 'ring20', 'ring21', 'ring22',
                                               'ring23', 'ring24', 'ring25', 'ring26', 'ring27', 'ring28', 'ring29',
                                               'ring30', 'ring31', 'ring32'])
        df['vzvpd'] = vz_vpd
        df['refmult3'] = refmult3
        df['protons'] = protons
        df['antiprotons'] = antiprotons
        df["net_protons"] = protons_net
        frames = [protons_df, df]
        protons_df = pd.concat(frames)
        protons_df.reset_index(drop=True, inplace=True)
        protons_df.to_pickle(file+"protons.pkl")

        print(run_num)

    except Exception as e:  # For any issues that might pop up.
        print(e.__class__, "occurred in", run_num)
        count += 1
        continue
    count += 1

print("Total events: ", event_total)
