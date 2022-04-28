import uproot3 as up
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
# up.default_library = "np"

# The purpose of this bit of code is to run over all the available
# 14.5 GeV, FastOffline data and get data for cuts on runs where things
# seems to not conform to what we would expect; things like a <V_z> very
# different than all of the other runs. This code will get the data, and
# then the averages and listing of the non-conforming runs will be done
# elsewhere.

# This is where our data lives and where we want to store the
# output arrays. This should be the only thing you have to edit
# in order to get the code up and running.
dataDirect = r"E:\2019Picos\14p5GeV\Runs"
saveDirect = r"C:\Users\dansk\Documents\Thesis\Protons/WIP/QA_Cuts"

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
RefMult_TOFMult = np.zeros((a, a))
RefMult_TOFMult_bins = np.zeros((2, a))
RefMult_TOFMatch = np.zeros((a, a))
RefMult_TOFMatch_bins = np.zeros((2, a))
RefMult_BetaEta = np.zeros((a, a))
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

# This is for the run ID (to identify runs to skip from QA).
Runs = []
# And it's nice to have a tally of the events you went over
# for the current analysis.
Events = 0

os.chdir(dataDirect)
r = len(os.listdir())
count = 1
print("Working on file:")
for file in os.listdir():
    # This is just to show how far along the script is.
    if count % 100 == 0:
        print(count, "of", r)
    # This is to make sure it's a ROOT file (it will error otherwise).
    if Path(file).suffix != '.root':
        continue
    # Use this to cut off the loop for testing.
    if count > 3:
        break
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

    # These are the event and track loops. It's written such that it
    # will ignore any iterations that are way "out of bounds." This shouldn't
    # be necessary, but it's here just in case. The stuff after "try" is
    # the loop proper, and the following "except" is for the really
    # high outliers ("IndexError" means the indicies don't match).
    # This will toss all the data from that run, not just the outlier that
    # caused the issue, so this is only for troubleshooting. If you did
    # your job right, you'll never see the error. If you see the error,
    # something needs to be corrected.
    try:
        Runs.append(file[:-5])

        # Event level quantities.
        vX = np.hstack(np.asarray(data["Event"]["Event.mPrimaryVertexX"].array()))
        vY = np.hstack(np.asarray(data["Event"]["Event.mPrimaryVertexY"].array()))
        vZ = np.hstack(np.asarray(data["Event"]["Event.mPrimaryVertexZ"].array()))
        vR = np.power((np.power(vX, 2)+np.power(vY, 2)), (1/2))
        ZDCx = np.hstack(np.asarray(data["Event"]["Event.mZDCx"].array()))
        Events += len(vX)

        RefMult3 = np.hstack(np.asarray(data["Event"]["Event.mRefMult3PosEast"].array()) +
                             np.asarray(data["Event"]["Event.mRefMult3PosWest"].array()) +
                             np.asarray(data["Event"]["Event.mRefMult3NegEast"].array()) +
                             np.asarray(data["Event"]["Event.mRefMult3NegWest"].array()))
        TOFMult = np.hstack(np.asarray(data["Event"]["Event.mbTofTrayMultiplicity"].array()))
        TOFMatch = np.hstack(np.asarray(data["Event"]["Event.mNBTOFMatch"].array()))

        # Track level quantities:
        oX = np.hstack(np.asarray(data["Track"]["Track.mOriginX"].array()))
        oY = np.hstack(np.asarray(data["Track"]["Track.mOriginY"].array()))
        oZ = np.hstack(np.asarray(data["Track"]["Track.mOriginZ"].array()))
        add_pi = np.hstack(np.where((oX < 0.0) & (oY > 0.0)))
        subtract_pi = np.hstack(np.where((oX < 0.0) & (oY < 0.0)))
        Phi = np.arctan(np.divide(oY, oX))
        Phi[add_pi] = Phi[add_pi] + np.pi
        Phi[subtract_pi] = Phi[subtract_pi] - np.pi

        pX = np.asarray(data["Track"]["Track.mGMomentumX"].array())
        pY = np.asarray(data["Track"]["Track.mGMomentumY"].array())
        pZ = np.asarray(data["Track"]["Track.mGMomentumZ"].array())
        pT = np.power((np.power(pX, 2) + np.power(pY, 2)), (1 / 2))
        pG = np.power((np.power(pX, 2) + np.power(pY, 2) + np.power(pZ, 2)), (1 / 2))
        eta_num = np.add(np.hstack(pG), np.hstack(pT))
        eta_div = np.subtract(np.hstack(pG), np.hstack(pT))
        eta_div[eta_div == 0.0] = 0.000000001  # To avoid infinities
        Eta = np.divide(np.log(np.divide(eta_num, eta_div)), 2)
        pZ_direction = np.hstack(np.where(np.hstack(pZ) < 0.0))
        Eta[pZ_direction] = -Eta[pZ_direction]

        DcaX = np.asarray(data["Track"]["Track.mOriginX"].array())-vX
        DcaY = np.asarray(data["Track"]["Track.mOriginY"].array())-vY
        DcaZ = np.asarray(data["Track"]["Track.mOriginZ"].array())-vZ
        Dca = np.power((np.power(DcaX, 2) + np.power(DcaY, 2) + np.power(DcaZ, 2)), (1/2))

        nHitsDedx = np.asarray(data["Track"]["Track.mNHitsDedx"].array())
        nHitsFit = np.asarray(data["Track"]["Track.mNHitsFit"].array())
        nHitsMax = np.asarray(data["Track"]["Track.mNHitsMax"].array())

        charge = fn.chargeList(nHitsFit)
        # Scaling for beta is a STAR thing; see StPicoBTofPidTraits.h.
        beta = np.asarray((data["BTofPidTraits"]["BTofPidTraits.mBTofBeta"].array())/20000.0)
        TofPid = np.asarray(data["BTofPidTraits"]["BTofPidTraits.mTrackIndex"].array())
        dEdX = np.asarray(data["Track"]["Track.mDedx"].array())

        # These are for histogramming with ToF matches.
        beta_eta1 = []
        pG_tof = []
        charge_tof = []
        dEdX_tof = []

        # ToF matching, mass, and histograms needing ToF.
        for i in range(int(len(vX))):
            try:
                index = TofPid[i]  # ToF match cut.
                pZ_tof = pZ[i][index]
                pZ_dir = np.hstack(np.where(pZ_tof < 0.0))
                pG_tof.append(pG[i][index])
                charge_tof.append(charge[i][index])
                dEdX_tof.append(dEdX[i][index])
                ETA_num = np.add(pG[i][index], pT[i][index])
                ETA_div = np.subtract(pG[i][index], pT[i][index])
                ETA_div[ETA_div == 0.0] = 0.000000001  # To avoid infinities
                ETA = np.divide(np.log(np.divide(ETA_num, ETA_div)), 2)
                ETA[pZ_dir] = -ETA[pZ_dir]
                beta_eta1_match = np.hstack(np.where((beta[i] > 0.1) & (np.absolute(ETA) < 1.0) & (np.absolute(Dca[i][index] < 3.0) & (nHitsFit[i][index] > 10))))
                beta_eta1.append(len(beta_eta1_match))

            except IndexError:  # For if there are no ToF matches in an event.
                print("Index error in track loop at", i)
                continue

        beta_eta1 = np.asarray(beta_eta1)
        charge = np.hstack(charge)
        pT = np.hstack(pT)
        pG = np.hstack(pG)
        dEdX = np.hstack(dEdX)
        nHitsFit = np.hstack(nHitsFit)
        nHitsDedx = np.hstack(nHitsDedx)
        Dca = np.hstack(Dca)
        pG_tof = np.hstack(np.asarray(pG_tof))
        charge_tof = np.hstack(np.asarray(charge_tof))
        dEdX_tof = np.hstack(np.asarray(dEdX_tof))
        beta = np.hstack(beta)

        p_squared = np.power(pG_tof, 2)
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

        # ToF and mass histogram filling.
        # m_squared vs p*q
        counter, binsX, binsY = np.histogram2d(m_squared, np.multiply(pG_tof, charge_tof),
                                               bins=a, range=((0, 1.5), (-5, 5)))
        m_pq += counter
        m_pq_bins = (binsX, binsY)

        # 1/beta vs p
        counter, binsX, binsY = np.histogram2d(beta, pG_tof, bins=a,
                                               range=((0.5, 3.6), (0, 10)))
        beta_p += counter
        beta_p_bins = (binsX, binsY)

        # dE/dX vs p*q
        counter, binsX, binsY = np.histogram2d(dEdX_tof, np.multiply(charge_tof, pG_tof), bins=a,
                                               range=((0, 31), (-3, 3)))
        dEdX_pq += counter
        dEdX_pq_bins = (binsX, binsY)

        # z vertex position
        counter, bins = np.histogram(vZ, bins=a, range=(-200, 200))
        v_z[0] += counter
        v_z[1] = bins[:-1]

        # transverse vertex position
        counter, binsX, binsY = np.histogram2d(
            vY, vX, bins=a, range=((-10, 10), (-10, 10)))
        v_r += counter
        v_r_bins = (binsX, binsY)

        # ToFMult vs RefMult
        counter, binsX, binsY = np.histogram2d(TOFMult, RefMult3, bins=a,
                                               range=((0, 1700), (0, 1000)))
        RefMult_TOFMult += counter
        RefMult_TOFMult_bins = (binsX, binsY)

        # nToFMatch vs RefMult3
        counter, binsX, binsY = np.histogram2d(TOFMatch, RefMult3, bins=a,
                                               range=((0, 500), (0, 1000)))
        RefMult_TOFMatch += counter
        RefMult_TOFMatch_bins = (binsX, binsY)

        # RefMult vs beta_eta1
        counter, binsX, binsY = np.histogram2d(beta_eta1, RefMult3, bins=a,
                                               range=((0, 100), (0, 1000)))
        RefMult_BetaEta += counter
        RefMult_BetaEta_bins = (binsX, binsY)

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
        counter, bins = np.histogram(nHitsFit*charge, bins=b, range=(-(b-1)/2, (b-1)/2))
        nHitsFit_charge[0] += counter
        nHitsFit_charge[1] = bins[:-1]

        # nHitsdEdX
        counter, bins = np.histogram(nHitsDedx, bins=c, range=(0, c-1))
        nHits_dEdX[0] += counter
        nHits_dEdX[1] = bins[:-1]

        # And that's the end of the loop.
        count += 1
    except IndexError:
        count += 1
        # Identifies the misbehaving file.
        print("IndexError in main loop at", file)
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
np.save("vZ.npy", vZ)
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
