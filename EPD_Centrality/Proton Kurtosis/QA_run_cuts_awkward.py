import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
import seaborn as sns
import functions as fn
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema as arex
from scipy.signal import savgol_filter as sgf
import uproot as up
import awkward as ak
import time

# This is just to find performance bottlenecks.
time_start = time.perf_counter()
time_keeper = []

# The purpose of this bit of code is to run over all the available
# 14.5 GeV, FastOffline data and get data to make cuts on runs where
# things seems to not conform to what we would expect; things like a
# <v_z> very different than all of the other runs. This code will get
# the data, and then the averages and listing of the non-conforming
# runs will be done elsewhere (currently: QA_Plots.py).

# This is where our data lives and where we want to store the
# output arrays. This should be the only thing you have to edit
# in order to get the code up and running.
dataDirect = r"E:\2019Picos\7p7GeV\Runs"
saveDirect = r"C:\Users\dansk\Documents\Thesis\Protons\7p7_GeV\QA"

# The following are all for the average value for an event.
# The second row is for the error.
AveRefMult3 = [[], []]
AveVz = [[], []]
AvePt = [[], []]
AveEta = [[], []]
AveVr = [[], []]
AveZdcX = [[], []]
AvePhi = [[], []]
AveDca = [[], []]

# Arrays to hold our histogram data. We need to have a separate array
# for the bins (this should go to pandas in the future as it's easier
# to keep track of, but alas we live in the present).
a = 1000
b = 161
c = 86
d = 101
v_z = np.zeros((2, a))  # bins included in this one
v_r = np.zeros((a, a))
v_r_bins = np.zeros((2, a))
RefMult_TOFMult = np.zeros((1700, a))
RefMult_TOFMult_bins = np.zeros((2, a))
RefMult_TOFMatch = np.zeros((a, a))
RefMult_TOFMatch_bins = np.zeros((2, a))
RefMult_BetaEta = np.zeros((400, a))
RefMult_BetaEta_bins = np.zeros((2, a))
p_t = np.zeros((2, a))  # bins included in this one
phi = np.zeros((2, a))  # bins included in this one
dca = np.zeros((2, a))  # bins included in this one
eta = np.zeros((2, a))  # bins included in this one
nHitsFit_charge = np.zeros((2, b))  # bins included in this one
nHits_dEdX = np.zeros((2, c))  # bins included in this one
m_pq = np.zeros((a, a))
m_pq_bins = np.zeros((2, a))
beta_p = np.zeros((a, a))
beta_p_bins = np.zeros((2, a))
dEdX_pq = np.zeros((a, a))
dEdX_pq_bins = np.zeros((2, a))
ref_sum = np.zeros((2, a))  # bins included in this one
ring_sum = np.zeros((2, a))  # bins included in this one

# This is to get the days for EPD 14.5 GeV calibration.
caliDays = np.asarray(("94", "105", "110", "113", "114", "123", "138", "139"))

# This is for the run ID (to identify runs to skip from QA).
Runs = []
# And it's nice to have a tally of the events you went over
# for the current analysis.
Events = 0

# Now to loop over the picos.
os.chdir(dataDirect)
r = len(os.listdir())
count = 1
print("Working on file:")
for file in sorted(os.listdir()):
    time_run = time.perf_counter()
    # This is just to show how far along the script is.
    if count % 20 == 0:
        print(count, "of", r)
    # This is to make sure it's a ROOT file (it will error otherwise).
    if Path(file).suffix != '.root':
        r -= 1
        continue
    # This is to get calibrated EPD nMIPs. These are 744 tiles, 0-743.
    try:
        fileLoc = r'D:\14GeV\ChiFit\Nmip_Day_'
        dayFit = 110  # file[2:5]  # <- That's to make it automatic; turned off for now.
        fileDay = np.where(caliDays.astype(int) < int(dayFit))[0][-1]
        nFits = np.loadtxt(fileLoc + caliDays[fileDay] + ".txt")[:, 4]
    except Exception as e:  # For any issues that might pop up.
        print(e.__class__, "occurred in reading EPD information.")
        count += 1
        continue
    # Use this to cut off the loop for testing.
    # if count > 5:
    #    break
    # Use this to run over a certain subset.
    # if count < 1200:
    #    count += 1
    #    continue
    # if count > 1205:
    #    break
    data = up.open(file)
    try:
        data = data["PicoDst"]
    except ValueError:  # Skip empty picos.
        r -= 1
        print("ValueError at", file)  # Identifies the misbehaving file.
        continue
    except KeyError:  # Skip non empty picos that have no data.
        r -= 1
        print("KeyError at", file)  # Identifies the misbehaving file.
        continue

    # This is the actual event and track "loops" (not really a loop as it's in Python).
    try:
        Runs.append(file[:-5])

        # Event level quantities.
        vX = ak.to_numpy(ak.flatten(data["Event"]["Event.mPrimaryVertexX"].array()))
        vY = ak.to_numpy(ak.flatten(data["Event"]["Event.mPrimaryVertexY"].array()))
        vZ = ak.to_numpy(ak.flatten(data["Event"]["Event.mPrimaryVertexZ"].array()))
        vR = np.power((np.power(vX, 2)+np.power(vY, 2)), (1/2))
        ZDCx = ak.to_numpy(ak.flatten(data["Event"]["Event.mZDCx"].array()))
        Events += len(vX)

        RefMult3 = ak.to_numpy(ak.flatten(data["Event"]["Event.mRefMult3PosEast"].array() +
                                          data["Event"]["Event.mRefMult3PosWest"].array() +
                                          data["Event"]["Event.mRefMult3NegEast"].array() +
                                          data["Event"]["Event.mRefMult3NegWest"].array()))
        TOFMult = ak.to_numpy(ak.flatten(data["Event"]["Event.mbTofTrayMultiplicity"].array()))
        TOFMatch = ak.to_numpy(ak.flatten(data["Event"]["Event.mNBTOFMatch"].array()))
        time2 = time.perf_counter() - time_run
        # print("Main entries done:", time2)
        # EPD ring information
        epdID = data["EpdHit"]["EpdHit.mId"].array()
        epd_mQTdata = data["EpdHit"]["EpdHit.mQTdata"].array()
        # nMIP = data["EpdHit"]["EpdHit.mnMIP"].array()  # Only for calibrated picos.
        time3 = time.perf_counter() - time2
        # print("EPD input started:", time3)
        ring_ref_row = fn.epdRing_awk(epdID)
        ring_ref_tile = fn.EPDTile_awk(epdID)
        #plt.hist(ak.flatten(ring_ref_row), bins=32, histtype='step')
        #plt.show()
        time4 = time.perf_counter() - time3
        # print("EPD input done:", time4)
        cal_val = ak.concatenate([nFits[np.newaxis]] * ak.size(ring_ref_tile, axis=0))
        # cal_val1 = ak.Array(np.tile(nFits, (Events, 1)))
        # print(cal_val == cal_val1)
        time5 = time.perf_counter() - time4
        # print("Intra EPD:", time5)
        cal_val = cal_val[ring_ref_tile]
        # This is to just use the outer rings.
        selector = ((ring_ref_row == 7) ^ (ring_ref_row == 8) ^ (ring_ref_row == 9) ^ (ring_ref_row == 10) ^
                    (ring_ref_row == 11) ^ (ring_ref_row == 12) ^ (ring_ref_row == 13) ^ (ring_ref_row == 14) ^
                    (ring_ref_row == 15) ^ (ring_ref_row == 23) ^ (ring_ref_row == 24) ^ (ring_ref_row == 25) ^
                    (ring_ref_row == 26) ^ (ring_ref_row == 27) ^ (ring_ref_row == 28) ^ (ring_ref_row == 29) ^
                    (ring_ref_row == 30) ^ (ring_ref_row == 31))
        epd_mQTdata = epd_mQTdata[selector]
        epd_adc = fn.epd_adc_awk(epd_mQTdata)
        epd_adc = epd_adc/cal_val[selector]
        epd_adc = ak.where(epd_adc > 3.0, 3.0, epd_adc)
        epd_adc = ak.where(epd_adc < 0.2, 0.0, epd_adc)
        EPDMult = ak.sum(epd_adc, axis=-1)
        ##########################################
        time_test = time.perf_counter() - time5
        # print("After EPD:", time_test)
        ##########################################
        # plt.hist(EPDMult, bins=100, histtype='step')
        # plt.yscale('log')
        # plt.show()
        # plt.hist(RefMult3, bins=100, histtype='step')
        # plt.yscale('log')
        # plt.show()
        # Track level quantities. I will ToF match all quantities.
        TofPid = data["BTofPidTraits"]["BTofPidTraits.mTrackIndex"].array()

        oX = data["Track"]["Track.mOriginX"].array()[TofPid]
        oY = data["Track"]["Track.mOriginY"].array()[TofPid]
        oZ = data["Track"]["Track.mOriginZ"].array()[TofPid]
        Phi = np.arctan2(oY, oX)

        pX = data["Track"]["Track.mGMomentumX"].array()[TofPid]
        pY = data["Track"]["Track.mGMomentumY"].array()[TofPid]
        pZ = data["Track"]["Track.mGMomentumZ"].array()[TofPid]
        pT = np.power((np.power(pX, 2) + np.power(pY, 2)), (1 / 2))
        pG = np.power((np.power(pX, 2) + np.power(pY, 2) + np.power(pZ, 2)), (1 / 2))
        Eta = np.arcsinh(np.divide(pZ, np.sqrt(np.add(np.power(pX, 2), np.power(pY, 2)))))

        DcaX = data["Track"]["Track.mOriginX"].array()[TofPid]-vX
        DcaY = data["Track"]["Track.mOriginY"].array()[TofPid]-vY
        DcaZ = data["Track"]["Track.mOriginZ"].array()[TofPid]-vZ
        Dca = np.power((np.power(DcaX, 2) + np.power(DcaY, 2) + np.power(DcaZ, 2)), (1/2))

        nHitsDedx = data["Track"]["Track.mNHitsDedx"].array()[TofPid]
        nHitsFit = data["Track"]["Track.mNHitsFit"].array()[TofPid]
        nHitsMax = data["Track"]["Track.mNHitsMax"].array()[TofPid]

        charge = fn.chargeList(nHitsFit)
        # Scaling for beta is a STAR thing; see StPicoBTofPidTraits.h.
        beta = data["BTofPidTraits"]["BTofPidTraits.mBTofBeta"].array()/20000.0
        dEdX = data["Track"]["Track.mDedx"].array()[TofPid]

        # These are for histogramming with ToF matches.
        beta_eta1_match = ak.where((beta >= 0.1) & (np.absolute(Eta) <= 1.0) & (np.absolute(Dca <= 3.0) &
                                    (np.absolute(nHitsFit) >= 10)), 1, 0)
        beta_eta1 = ak.sum(beta_eta1_match, axis=-1)

        charge = ak.to_numpy(ak.flatten(charge))
        pT = ak.to_numpy(ak.flatten(pT))
        pG = ak.to_numpy(ak.flatten(pG))
        dEdX = ak.to_numpy(ak.flatten(dEdX))
        nHitsFit = ak.to_numpy(ak.flatten(nHitsFit))
        nHitsDedx = ak.to_numpy(ak.flatten(nHitsDedx))
        Dca = ak.to_numpy(ak.flatten(Dca))
        beta = ak.to_numpy(ak.flatten(beta))
        Phi = ak.to_numpy(ak.flatten(Phi))
        Eta = ak.to_numpy(ak.flatten(Eta))
        beta_eta1 = ak.to_numpy(beta_eta1)
        EPDMult = ak.to_numpy(EPDMult)

        p_squared = np.power(pG, 2)
        b_squared = np.power(beta, 2)
        b_squared[b_squared == 0.0] = 1e-10  # to avoid infinities
        g_squared = (1 - np.power(beta, 2))
        m_squared = np.divide(np.multiply(p_squared, g_squared), b_squared)

        beta[beta < 0.001] = 0.001  # To avoid infinities
        beta = np.power(beta, -1)  # For graphing

        # Event level average values.
        AveRefMult3[0].append(np.mean(RefMult3))
        AveRefMult3[1].append(np.divide(np.std(RefMult3), len(RefMult3)))
        AveVz[0].append(np.mean(vZ))
        AveVz[1].append(np.divide(np.std(vZ), len(vZ)))
        AveVr[0].append(np.mean(vR))
        AveVr[1].append(np.divide(np.std(vR), len(vR)))
        AveZdcX[0].append(np.mean(ZDCx))
        AveZdcX[1].append(np.divide(np.std(ZDCx), len(ZDCx)))

        # Track level average values.
        AveEta[0].append(np.mean(Eta))
        AveEta[1].append(np.divide(np.std(Eta), len(Eta)))
        AvePhi[0].append(np.mean(Phi))
        AvePhi[1].append(np.divide(np.std(Phi), len(Phi)))
        AveDca[0].append(np.mean(Dca))
        AveDca[1].append(np.divide(np.std(Dca), len(Dca)))
        AvePt[0].append(np.mean(pT))
        AvePt[1].append(np.divide(np.std(pT), len(pT)))

        # Now to fill our histograms.

        # 1D histograms for RefMult and EPDMult
        counter, bins = np.histogram(RefMult3, bins=a, range=(0, a))
        ref_sum[0] += counter
        ref_sum[1] = bins[:-1]

        counter, bins = np.histogram(EPDMult, bins=a, range=(0, a))
        ring_sum[0] += counter
        ring_sum[1] = bins[:-1]

        # ToF and mass histogram filling.
        # m_squared vs p*q
        counter, binsX, binsY = np.histogram2d(m_squared, np.multiply(pG, charge),
                                               bins=a, range=((0, 1.5), (-5, 5)))
        m_pq += counter
        m_pq_bins = (binsX, binsY)

        # 1/beta vs p
        counter, binsX, binsY = np.histogram2d(beta, pG, bins=a,
                                               range=((0.5, 3.6), (0, 10)))
        beta_p += counter
        beta_p_bins = (binsX, binsY)

        # dE/dX vs p*q
        counter, binsX, binsY = np.histogram2d(dEdX, np.multiply(charge, pG), bins=a,
                                               range=((0, 31), (-3, 3)))
        dEdX_pq += counter
        dEdX_pq_bins = (binsX, binsY)

        # z vertex position
        counter, bins = np.histogram(vZ, bins=a, range=(-200, 200))
        v_z[0] += counter
        v_z[1] = bins[:-1]

        # transverse vertex position
        counter, binsX, binsY = np.histogram2d(vY, vX, bins=a,
                                               range=((-10, 10), (-10, 10)))
        v_r += counter
        v_r_bins = (binsX, binsY)

        # ToFMult vs RefMult
        counter, binsX, binsY = np.histogram2d(TOFMult, RefMult3, bins=(1700, a),
                                               range=((0, 1700), (0, 1000)))
        RefMult_TOFMult += counter
        RefMult_TOFMult_bins = np.array((binsX, binsY), dtype=object)

        # nToFMatch vs RefMult3
        counter, binsX, binsY = np.histogram2d(TOFMatch, RefMult3, bins=a,
                                               range=((0, 500), (0, 1000)))
        RefMult_TOFMatch += counter
        RefMult_TOFMatch_bins = (binsX, binsY)

        # RefMult vs beta_eta1
        counter, binsX, binsY = np.histogram2d(beta_eta1, RefMult3,
                                               bins=(400, a), range=((0, 400), (0, 1000)))
        RefMult_BetaEta += counter
        RefMult_BetaEta_bins = np.array((binsX, binsY), dtype=object)

        # Event and track level histograms (no ToF stuff)
        # Transverse momentum
        counter, bins = np.histogram(pT, bins=a, range=(0, 6))
        p_t[0] += counter
        p_t[1] = bins[:-1]

        # phi
        counter, bins = np.histogram(Phi, bins=a, range=(-np.pi - 0.2, np.pi + 0.2))
        phi[0] += counter
        phi[1] = bins[:-1]

        # DCA
        counter, bins = np.histogram(Dca, bins=a, range=(0, 5))
        dca[0] += counter
        dca[1] = bins[:-1]

        # eta
        counter, bins = np.histogram(Eta, bins=a, range=(-3, 3))
        eta[0] += counter
        eta[1] = bins[:-1]

        # nHitsFit*q
        counter, bins = np.histogram(nHitsFit, bins=b, range=(-(b-1)/2, (b-1)/2))
        nHitsFit_charge[0] += counter
        nHitsFit_charge[1] = bins[:-1]

        # nHitsdEdX
        counter, bins = np.histogram(nHitsDedx, bins=c, range=(0, c-1))
        nHits_dEdX[0] += counter
        nHits_dEdX[1] = bins[:-1]

        # And that's the end of the loop.
        time_keeper.append(time.perf_counter() - time_run)
        # print("Final:", time.perf_counter()-time_test)
        count += 1
    except IndexError:
        count += 1
        # Identifies the misbehaving file.
        print("IndexError in main loop at", file)
        continue
    except Exception as e:  # For any issues that might pop up.
        print(e.__class__, "occurred in the main loop at", file)
        count += 1
        continue

print("All files analysed.")
print("Total events:", Events)

# Let's save where we want to save.
os.chdir(saveDirect)

# First to save the average values per run.
AveRefMult3 = np.asarray(AveRefMult3)
AveVz = np.asarray(AveVz)
AvePt = np.asarray(AvePt)
AvePhi = np.asarray(AvePt)
AveVr = np.asarray(AveVr)
AveZdcX = np.asarray(AveZdcX)
AveEta = np.asarray(AveEta)
AveDca = np.asarray(AveDca)
# Let's save it in Pandas.
df = pd.DataFrame({
    'RefMult3': AveRefMult3[0],
    'RefMult3_err': AveRefMult3[1],
    'vZ': AveVz[0],
    'vZ_err': AveVz[1],
    'pT': AvePt[0],
    'pT_err': AvePt[1],
    'phi': AvePhi[0],
    'phi_err': AvePhi[1],
    'vR': AveVr[0],
    'vR_err': AveVr[1],
    'ZDCx': AveZdcX[0],
    'ZDCx_err': AveZdcX[1],
    'eta': AveEta[0],
    'eta_err': AveEta[1],
    'DCA': AveDca[0],
    'DCA_err': AveDca[1]
})
df.to_pickle("averages.pkl")

# Now to save the list of runs.
Runs = np.asarray(Runs)
np.save("run_list.npy", Runs)

# Now to save our histograms for reconstruction.
np.save("vZ.npy", v_z)
np.save("vR.npy", v_r)
np.save("vR_bins.npy", v_r_bins)
np.save("refmult_tofmult.npy", RefMult_TOFMult)
np.save("refmult_tofmult_bins.npy", RefMult_TOFMult_bins)
np.save("refmult_tofmatch.npy", RefMult_TOFMatch)
np.save("refmult_tofmatch_bins.npy", RefMult_TOFMatch_bins)
np.save("refmult_beta_eta.npy", RefMult_BetaEta)
np.save("refmult_beta_eta_bins.npy", RefMult_BetaEta_bins)
np.save("m_pq.npy", m_pq)
np.save("m_pq_bins.npy", m_pq_bins)
np.save("beta_p.npy", beta_p)
np.save("beta_p_bins.npy", beta_p_bins)
np.save("dEdX_pq.npy", dEdX_pq)
np.save("dEdX_pq_bins.npy", dEdX_pq_bins)
np.save("pT.npy", p_t)
np.save("phi.npy", phi)
np.save("dca.npy", dca)
np.save("eta.npy", eta)
np.save("nhitsfit_charge.npy", nHitsFit_charge)
np.save("nhits_dedx.npy", nHits_dEdX)
np.save("ref_sum.npy", ref_sum)
np.save("ring_sum.npy", ring_sum)

'''
# Just more performance assessment.
np.asarray(time_keeper)
time_final = time.perf_counter() - time_start
print("Total run time:", time_final)
print("Average time per run:", np.mean(time_keeper))
print("Longest run:", Runs[np.argmax(time_keeper)], "time:", np.max(time_keeper))
'''
