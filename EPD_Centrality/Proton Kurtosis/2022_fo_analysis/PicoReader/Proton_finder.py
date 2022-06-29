#
# \PicoDst reader for Python
#
# \author Skipper Kagamaster
# \date 03/19/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
# /

"""
This code is for finding protons by making cuts from the "bad" runs
and some QA parameters (outlined in the code proper).

The directories are where our data lives and where we want to store the
output arrays. This should be the only thing you have to edit in order
to get the code up and running.
dataDirect is for the picos to run over
saveDirect is where the bad run list is stored
finalDirect is for what this code finds

The output will be arrays of the following:
RunID, events, RefMult3, protons, antiprotons, EPD rings
All will be eventwise correlated. This makes for a large,
final data set, but then all the statistical work can be
performed elsewhere (with the benefit of preserving all
correlations should you want to change things up, like
centrality cuts or associations between EPD ring fits).
"""
import copy
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from pathlib import Path
import pico_reader as rdr
from scipy.signal import savgol_filter as sgf
import uproot as up
import awkward as ak

dataDirect = r"F:\2019Picos\14p5GeV\Runs"
saveDirect = r"D:\14GeV\Thesis\PythonArrays"

os.chdir(saveDirect)
bad_runs = np.load("badruns.npy", allow_pickle=True)
"""
We need to make one more cut: average protons. If this is already included
in badruns.npy, ignore this bit.
"""
proton_ave = [[[], []], [[], []]]  # For high and low p ranges, p and p-
proton_std = [[[], []], [[], []]]
runlist = []
"""
Arrays to hold our histogram data.
"""
# TODO Make histogramming a lot more concise; this is a mess.
a = 1000
b = 160
c = 85
d = 101
###########################################
# ------------ 1D histograms ------------ #
###########################################
# Z vertex position
v_z_counts, v_z_bins = np.histogram(0, a, range=(-100, 100))
# Transverse momentum
p_t_count, p_t_bins = np.histogram(0, bins=a, range=(0, 6))
# Azimuthal angle
phi_count, phi_bins = np.histogram(0, bins=a, range=(-np.pi - 0.2, np.pi + 0.2))
# Distance of closest approach
dca_count, dca_bins = np.histogram(0, bins=a, range=(0, 5))
# Pseudorapidity
eta_count, eta_bins = np.histogram(0, bins=a, range=(-3, 3))
# Rapidity (based on proton mass)
rap_count, rap_bins = np.histogram(0, bins=a, range=(-6, 6))
# Number of hits in TPC * charge value
nhq_count, nhq_bins = np.histogram(0, bins=b, range=(-80, 80))
# Number of hits used to construct dE/dx
nhde_count, nhde_bins = np.histogram(0, bins=c, range=(0, 85))
# "Ratio" of NHits to NhitsMax (see construction)
nhr_count, nhr_bins = np.histogram(0, bins=200, range=(0, 5))
# Z vertex position (after cuts)
av_z_counts, av_z_bins = np.histogram(0, a, range=(-100, 100))
# Transverse momentum (after cuts)
ap_t_count, ap_t_bins = np.histogram(0, bins=a, range=(0, 6))
# Azimuthal angle (after cuts)
aphi_count, aphi_bins = np.histogram(0, bins=a, range=(-np.pi - 0.2, np.pi + 0.2))
# Distance of closest approach (after cuts)
adca_count, adca_bins = np.histogram(0, bins=a, range=(0, 5))
# Pseudorapidity (after cuts)
aeta_count, aeta_bins = np.histogram(0, bins=a, range=(-3, 3))
# Rapidity (based on proton mass) (after cuts)
arap_count, arap_bins = np.histogram(0, bins=a, range=(-6, 6))
# Number of hits in TPC * charge value (after cuts)
anhq_count, anhq_bins = np.histogram(0, bins=b, range=(-80, 80))
# Number of hits used to construct dE/dx (after cuts)
anhde_count, anhde_bins = np.histogram(0, bins=c, range=(0, 85))
# "Ratio" of NHits to NhitsMax (see construction) (after cuts)
anhr_count, anhr_bins = np.histogram(0, bins=200, range=(0, 5))
###########################################
# ------------ 2D histograms ------------ #
###########################################
# Transverse vertex position
v_r_counts, v_r_binsX, v_r_binsY = np.histogram2d([0], [0], bins=a, range=((-10, 10), (-10, 10)))
# RefMult3 vs ToFMult
rt_mult_counts, rt_mult_binsX, rt_mult_binsY = np.histogram2d([0], [0], bins=(a, 1700),
                                                              range=((0, 1000), (0, 1700)))
# RefMult3 vs ToF matched tracks
rt_match_count, rt_match_binsX, rt_match_binsY = np.histogram2d([0], [0], bins=a,
                                                                range=((0, 1000), (0, 500)))
# RefMult3 vs BetaEta1 (see construction)
rb_count, rb_binsX, rb_binsY = np.histogram2d([0], [0], bins=(1000, 400),
                                              range=((0, 1000), (0, 400)))
# m^2 vs p*q
mpq_count, mpq_binsX, mpq_binsY = np.histogram2d([0], [0], bins=a, range=((-5, 5), (0, 1.5)))
# 1/beta vs momentum
bp_count, bp_binsX, bp_binsY = np.histogram2d([0], [0], bins=a, range=((0.5, 3.6), (0, 10)))
# dE/dx vs p*q
dEp_count, dEp_binsX, dEp_binsY = np.histogram2d([0], [0], bins=a, range=((-3, 3), (0, 31)))
# Transverse vertex position (after cuts)
av_r_counts, av_r_binsX, av_r_binsY = np.histogram2d([0], [0], bins=a, range=((-10, 10), (-10, 10)))
# RefMult3 vs ToFMult (after cuts)
art_mult_counts, art_mult_binsX, art_mult_binsY = np.histogram2d([0], [0], bins=(a, 1700),
                                                                 range=((0, 1000), (0, 1700)))
# RefMult3 vs ToF matched tracks (after cuts)
art_match_count, art_match_binsX, art_match_binsY = np.histogram2d([0], [0], bins=a,
                                                                   range=((0, 1000), (0, 500)))
# RefMult3 vs BetaEta1 (see construction) (after cuts)
arb_count, arb_binsX, arb_binsY = np.histogram2d([0], [0], bins=(1000, 400),
                                                 range=((0, 1000), (0, 400)))
# m^2 vs p*q (after cuts)
ampq_count, ampq_binsX, ampq_binsY = np.histogram2d([0], [0], bins=a, range=((-5, 5), (0, 1.5)))
# 1/beta vs momentum (after cuts)
abp_count, abp_binsX, abp_binsY = np.histogram2d([0], [0], bins=a, range=((0.5, 3.6), (0, 10)))
# dE/dx vs p*q (after cuts)
adEp_count, adEp_binsX, adEp_binsY = np.histogram2d([0], [0], bins=a, range=((-3, 3), (0, 31)))
# m^2 vs p*q (after selection)
pmpq_count, pmpq_binsX, pmpq_binsY = np.histogram2d([0], [0], bins=a, range=((-5, 5), (0, 1.5)))
# 1/beta vs momentum (after selection)
pbp_count, pbp_binsX, pbp_binsY = np.histogram2d([0], [0], bins=a, range=((0.5, 3.6), (0, 10)))
# dE/dx vs p*q (after selection)
pdEp_count, pdEp_binsX, pdEp_binsY = np.histogram2d([0], [0], bins=a, range=((-3, 3), (0, 31)))
"""
These are to load the files for 14.5 GeV nMIP calibration for the EPD in FastOffline.
"""
cal_set = np.asarray([94, 105, 110, 113, 114, 123, 138, 139])
cal_files = []
for i in cal_set:
    cal_files.append(np.loadtxt(r'D:\14GeV\ChiFit\adc{}.txt'.format(i), delimiter=','))
"""
And now to load up our nSigmaProton mean shift calibrations, separated into
bins in p_T in 0.1 increments from 0.1 to 1.2 (11 bins). run_cal is the
corresponding run for each 11D entry in nSigCal.
"""
nSigCal = np.load(r'D:\14GeV\Thesis\PythonArrays\nSigmaCal.npy', allow_pickle=True)
run_cal = np.load(r'D:\14GeV\Thesis\PythonArrays\runs.npy', allow_pickle=True)

"""
Let's run it!
"""
os.chdir(dataDirect)
r = len(os.listdir())
files = []
for i in os.listdir():
    if i.endswith('root'):
        files.append(i)
count = 0
print("Working on file:")
for file in sorted(files):
    # This is to omit all runs marked "bad."
    run_num = int(file[:8])
    if run_num in bad_runs:
        r -= 1
        continue
    # This is just to show how far along the script is.
    if count % 20 == 0:
        print(count + 1, "of", r)
    data = up.open(file)
    try:
        # Here's the array to hold RefMult3, protons, antiprotons, and EPD eta ring sums.
        data_array = []
        for i in range(35):
            data_array.append([])
        day = int(file[2:5])
        cal_index = np.where(cal_set <= day)[0][-1]
        epd_cal = cal_files[cal_index]
        nsig_cal_ind = np.argwhere(run_cal == run_num)
        # Running for non_TOF matched and TOF-matched.
        # TODO: Make this more elegant.
        pico_low = rdr.PicoDST()
        nSigEnter = nSigCal[nsig_cal_ind][0][0]
        pico_low.import_data(file, cal_file=epd_cal, nsig_cals=nSigEnter)

        # ******** HISTOGRAM FILLING ********
        v_z_counts += np.histogram(pico_low.v_z, a, range=(-100, 100))[0]
        p_t_count += np.histogram(ak.to_numpy(ak.flatten(pico_low.p_t)), bins=a, range=(0, 6))[0]
        phi_count += np.histogram(ak.to_numpy(ak.flatten(pico_low.phi)), bins=a, range=(-np.pi - 0.2, np.pi + 0.2))[0]
        dca_count += np.histogram(ak.to_numpy(ak.flatten(pico_low.dca)), bins=a, range=(0, 5))[0]
        eta_count += np.histogram(ak.to_numpy(ak.flatten(pico_low.eta)), bins=a, range=(-3, 3))[0]
        rap_count += np.histogram(ak.to_numpy(ak.flatten(pico_low.rapidity)), bins=a, range=(-6, 6))[0]
        nhq_count += np.histogram(ak.to_numpy(ak.flatten(pico_low.nhitsfit)),
                                  bins=b, range=(-80, 80))[0]
        nhde_count += np.histogram(ak.to_numpy(ak.flatten(pico_low.nhitsdedx)), bins=c, range=(0, 85))[0]
        nhr_count += np.histogram(np.divide(1 + np.absolute(ak.to_numpy(ak.flatten(pico_low.nhitsfit))),
                                            1 + ak.to_numpy(ak.flatten(pico_low.nhitsmax))),
                                  bins=200, range=(0, 5))[0]
        v_r_counts += np.histogram2d(pico_low.v_x, pico_low.v_y, bins=a, range=((-10, 10), (-10, 10)))[0]
        rt_mult_counts += np.histogram2d(pico_low.refmult3, pico_low.tofmult,
                                         bins=(a, 1700), range=((0, 1000), (0, 1700)))[0]
        rt_match_count += np.histogram2d(pico_low.refmult3, pico_low.tofmatch,
                                         bins=a, range=((0, 1000), (0, 500)))[0]
        rb_count += np.histogram2d(pico_low.refmult3, pico_low.beta_eta_1,
                                   bins=(1000, 400), range=((0, 1000), (0, 400)))[0]
        tof_ind = pico_low.tofpid
        mpq_count += np.histogram2d(ak.to_numpy(ak.flatten(pico_low.p_g[tof_ind] * pico_low.charge[tof_ind])),
                                    ak.to_numpy(ak.flatten(pico_low.m_2)),
                                    bins=a, range=((-5, 5), (0, 1.5)))[0]
        beta = ak.where(pico_low.beta == 0, 1e-10, pico_low.beta)
        bp_count += np.histogram2d(ak.to_numpy(ak.flatten(pico_low.p_g[tof_ind])),
                                   1 / ak.to_numpy(ak.flatten(beta)),
                                   bins=a, range=((0.5, 3.6), (0, 10)))[0]
        dEp_count += np.histogram2d(ak.to_numpy(ak.flatten(pico_low.charge * pico_low.p_g)),
                                    ak.to_numpy(ak.flatten(pico_low.dedx)), bins=a, range=((-3, 3), (0, 31)))[0]
        # ******** END OF HISTOGRAM FILLING ********

        # Event cuts
        pico_low.event_cuts()
        pico_high = copy.deepcopy(pico_low)
        # Track cuts
        pico_low.track_qa_cuts()
        pico_high.track_qa_cuts_tof()

        # ******** HISTOGRAM FILLING ********
        av_z_counts += np.histogram(pico_low.v_z, a, range=(-100, 100))[0]
        ap_t_count += np.histogram(ak.to_numpy(ak.flatten(pico_low.p_t)), bins=a, range=(0, 6))[0]
        ap_t_count += np.histogram(ak.to_numpy(ak.flatten(pico_high.p_t)), bins=a, range=(0, 6))[0]
        aphi_count += np.histogram(ak.to_numpy(ak.flatten(pico_low.phi)), bins=a, range=(-np.pi - 0.2, np.pi + 0.2))[0]
        aphi_count += np.histogram(ak.to_numpy(ak.flatten(pico_high.phi)), bins=a, range=(-np.pi - 0.2, np.pi + 0.2))[0]
        adca_count += np.histogram(ak.to_numpy(ak.flatten(pico_low.dca)), bins=a, range=(0, 5))[0]
        adca_count += np.histogram(ak.to_numpy(ak.flatten(pico_high.dca)), bins=a, range=(0, 5))[0]
        aeta_count += np.histogram(ak.to_numpy(ak.flatten(pico_low.eta)), bins=a, range=(-3, 3))[0]
        aeta_count += np.histogram(ak.to_numpy(ak.flatten(pico_high.eta)), bins=a, range=(-3, 3))[0]
        arap_count += np.histogram(ak.to_numpy(ak.flatten(pico_low.rapidity)), bins=a, range=(-6, 6))[0]
        arap_count += np.histogram(ak.to_numpy(ak.flatten(pico_high.rapidity)), bins=a, range=(-6, 6))[0]
        anhq_count += np.histogram(ak.to_numpy(ak.flatten(pico_low.nhitsfit)),
                                   bins=b, range=(-80, 80))[0]
        anhq_count += np.histogram(ak.to_numpy(ak.flatten(pico_high.nhitsfit)),
                                   bins=b, range=(-80, 80))[0]
        anhde_count += np.histogram(ak.to_numpy(ak.flatten(pico_low.nhitsdedx)), bins=c, range=(0, 85))[0]
        anhde_count += np.histogram(ak.to_numpy(ak.flatten(pico_high.nhitsdedx)), bins=c, range=(0, 85))[0]
        anhr_count += np.histogram(np.divide(1 + np.absolute(ak.to_numpy(ak.flatten(pico_low.nhitsfit))),
                                             1 + ak.to_numpy(ak.flatten(pico_low.nhitsmax))),
                                   bins=200, range=(0, 5))[0]
        anhr_count += np.histogram(np.divide(1 + np.absolute(ak.to_numpy(ak.flatten(pico_high.nhitsfit))),
                                             1 + ak.to_numpy(ak.flatten(pico_high.nhitsmax))),
                                   bins=200, range=(0, 5))[0]
        av_r_counts += np.histogram2d(pico_low.v_x, pico_low.v_y, bins=a, range=((-10, 10), (-10, 10)))[0]
        art_mult_counts += np.histogram2d(pico_low.refmult3, pico_low.tofmult,
                                          bins=(a, 1700), range=((0, 1000), (0, 1700)))[0]
        art_match_count += np.histogram2d(pico_low.refmult3, pico_low.tofmatch,
                                          bins=a, range=((0, 1000), (0, 500)))[0]
        arb_count += np.histogram2d(pico_low.refmult3, pico_low.beta_eta_1,
                                    bins=(1000, 400), range=((0, 1000), (0, 400)))[0]
        ampq_count += np.histogram2d(ak.to_numpy(ak.flatten(pico_high.p_g * pico_high.charge)),
                                     ak.to_numpy(ak.flatten(pico_high.m_2)),
                                     bins=a, range=((-5, 5), (0, 1.5)))[0]
        abeta = ak.where(pico_high.beta == 0, 1e-10, pico_high.beta)
        abp_count += np.histogram2d(ak.to_numpy(ak.flatten(pico_high.p_g)), 1 / ak.to_numpy(ak.flatten(abeta)),
                                    bins=a, range=((0.5, 3.6), (0, 10)))[0]
        adEp_count += np.histogram2d(ak.to_numpy(ak.flatten(pico_low.charge * pico_low.p_g)),
                                     ak.to_numpy(ak.flatten(pico_low.dedx)), bins=a, range=((-3, 3), (0, 31)))[0]
        adEp_count += np.histogram2d(ak.to_numpy(ak.flatten(pico_high.charge * pico_high.p_g)),
                                     ak.to_numpy(ak.flatten(pico_high.dedx)), bins=a, range=((-3, 3), (0, 31)))[0]
        # ******** END OF HISTOGRAM FILLING ********

        # PID cuts
        pico_low.select_protons_low()
        pico_high.select_protons_high()

        # ******** HISTOGRAM FILLING ********
        pmpq_count += np.histogram2d(ak.to_numpy(ak.flatten(pico_high.m_2)),
                                     ak.to_numpy(ak.flatten(pico_high.p_g * pico_high.charge)),
                                     bins=a, range=((-5, 5), (0, 1.5)))[0]
        pbeta = ak.where(pico_high.beta == 0, 1e-10, pico_high.beta)
        pbp_count += np.histogram2d(ak.to_numpy(ak.flatten(pico_high.p_g)),1 / ak.to_numpy(ak.flatten(pbeta)),
                                    bins=a, range=((0.5, 3.6), (0, 10)))[0]
        pdEp_count += np.histogram2d(ak.to_numpy(ak.flatten(pico_low.charge * pico_low.p_g)),
                                     ak.to_numpy(ak.flatten(pico_low.dedx)), bins=a, range=((-3, 3), (0, 31)))[0]
        pdEp_count += np.histogram2d(ak.to_numpy(ak.flatten(pico_high.charge * pico_high.p_g)),
                                     ak.to_numpy(ak.flatten(pico_high.dedx)), bins=a, range=((-3, 3), (0, 31)))[0]
        # ******** END OF HISTOGRAM FILLING ********

        proton_ave[0][0].append(np.mean(pico_low.protons))
        proton_ave[0][1].append(np.mean(pico_low.antiprotons))
        proton_ave[1][0].append(np.mean(pico_high.protons))
        proton_ave[1][1].append(np.mean(pico_high.antiprotons))
        proton_std[0][0].append(np.std(pico_low.protons) / np.sqrt(len(pico_low.protons)))
        proton_std[0][1].append(np.std(pico_low.antiprotons) / np.sqrt(len(pico_low.antiprotons)))
        proton_std[1][0].append(np.std(pico_high.protons) / np.sqrt(len(pico_high.protons)))
        proton_std[1][1].append(np.std(pico_high.antiprotons) / np.sqrt(len(pico_high.antiprotons)))

        data_array[0] = np.hstack((data_array[0], np.asarray(pico_low.protons + pico_high.protons)))
        data_array[1] = np.hstack((data_array[1], np.asarray(pico_low.antiprotons + pico_high.antiprotons)))
        data_array[2] = np.hstack((data_array[2], pico_low.refmult3))
        for i in range(32):
            data_array[i+3] = np.hstack((data_array[i+3], np.asarray(pico_low.epd_hits[i])))
        data_array = np.asarray(data_array)
        # And save it for later analysis.
        np.save(saveDirect + r'\Analysis_Proton_Arrays' + r'\{}_protons.npy'.format(run_num),
                data_array)

        runlist.append(pico_low.run_id)
        count += 1
    except ValueError:  # Skip empty picos.
        r -= 1
        print("Run number", run_num, "is empty.")  # Identifies the misbehaving file.
        continue
    except KeyError:  # Skip non-empty picos that have no data.
        r -= 1
        print("Run number", run_num, "has no data.")  # Identifies the misbehaving file.
        continue
    except Exception as e:  # For any other issues that might pop up.
        print("Darn it!", e.__class__, "occurred in run", run_num)
        r -= 1
        continue
print("All files analysed.")

proton_ave = np.asarray(proton_ave)
proton_std = np.asarray(proton_std)
runlist = np.asarray(runlist)

# Let's save where we want to save.
saveDirect = saveDirect + r'\Analysis_Histograms'
os.chdir(saveDirect)
np.save('proton_ave.npy', proton_ave)
np.save('proton_std.npy', proton_std)
np.save('runlist.npy', runlist)

# Deary me, here come the histograms ...
np.save('v_z.npy', v_z_counts)
np.save('v_z_bins.npy', v_z_bins)
np.save('p_t.npy', p_t_count)
np.save('p_t_bins.npy', p_t_bins)
np.save('phi.npy', phi_count)
np.save('phi_bins.npy', phi_bins)
np.save('dca.npy', dca_count)
np.save('dca_bins.npy', dca_bins)
np.save('eta.npy', eta_count)
np.save('eta_bins.npy', eta_bins)
np.save('rap.npy', rap_count)
np.save('rap_bins.npy', rap_bins)
np.save('nhq.npy', nhq_count)
np.save('nhq_bins.npy', nhq_bins)
np.save('nhde.npy', nhde_count)
np.save('nhde_bins.npy', nhde_bins)
np.save('nhr.npy', nhr_count)
np.save('nhr_bins.npy', nhr_bins)
np.save('av_z.npy', av_z_counts)
np.save('ap_t.npy', ap_t_count)
np.save('aphi.npy', aphi_count)
np.save('adca.npy', adca_count)
np.save('aeta.npy', aeta_count)
np.save('arap.npy', arap_count)
np.save('anhq.npy', anhq_count)
np.save('anhde.npy', anhde_count)
np.save('anhr.npy', anhr_count)

np.save('v_r.npy', v_r_counts)
np.save('v_r_binsX.npy', v_r_binsX)
np.save('v_r_binsY.npy', v_r_binsY)
np.save('rt_mult.npy', rt_mult_counts)
np.save('rt_mult_binsX.npy', rt_mult_binsX)
np.save('rt_mult_binsY.npy', rt_mult_binsY)
np.save('rt_match.npy', rt_match_count)
np.save('rt_match_binsX.npy', rt_match_binsX)
np.save('rt_match_binsY.npy', rt_match_binsY)
np.save('rb.npy', rb_count)
np.save('rb_binsX.npy', rb_binsX)
np.save('rb_binsY.npy', rb_binsY)
np.save('mpq.npy', mpq_count)
np.save('mpq_binsX.npy', mpq_binsX)
np.save('mpq_binsY.npy', mpq_binsY)
np.save('bp.npy', bp_count)
np.save('bp_binsX.npy', bp_binsX)
np.save('bp_binsY.npy', bp_binsY)
np.save('dEp.npy', dEp_count)
np.save('dEp_binsX.npy', dEp_binsX)
np.save('dEp_binsY.npy', dEp_binsY)
np.save('av_r.npy', av_r_counts)
np.save('art_mult.npy', art_mult_counts)
np.save('art_match.npy', art_match_count)
np.save('arb.npy', arb_count)
np.save('ampq.npy', ampq_count)
np.save('abp.npy', abp_count)
np.save('adEp.npy', adEp_count)
np.save('pmpq.npy', pmpq_count)
np.save('pbp.npy', pbp_count)
np.save('pdEp.npy', pdEp_count)

# Example plot using dE/dx
X, Y = np.meshgrid(dEp_binsX, dEp_binsY)
plt.pcolormesh(X, Y, dEp_count.T, norm=LogNorm(), cmap='bone')
plt.pcolormesh(X, Y, pdEp_count.T, norm=LogNorm(), cmap='jet')
plt.colorbar()
plt.show()
