import os
import numpy as np
from pathlib import Path
import functions as fn
from scipy.signal import savgol_filter as sgf
import uproot as up
import awkward as ak

# This code is to find protons by making cuts from the "bad" runs
# and some QA parameters (outlined in the code proper).
# ----- THIS IS USING ONLY YU'S CUTS!! -----
#
# This is where our data lives and where we want to store the
# output arrays. This should be the only thing you have to edit
# in order to get the code up and running.
# dataDirect is for the picos to run over, and saveDirect is where
# the bad run list is stored. finalDirect is for what this code
# finds.
dataDirect = r"E:\2019Picos\14p5GeV\Runs"
saveDirect = r"C:\Users\dansk\Documents\Thesis\Protons\WIP\QA_Cuts"
finalDirect = r"C:\Users\dansk\Documents\Thesis\Protons\WIP\Protons\YuCuts"

os.chdir(saveDirect)
bad_runs = np.load("badRunsYu.npy", allow_pickle=True)

# We can make one more cut: average protons.
Ave_proton = [[], []]

# Yu's cuts for RefMult3:
RefCuts = np.asarray((10, 21, 41, 72, 118, 182, 270, 392, 472))

# This is to save our protons.
proton_counts = []
for i in range(len(RefCuts)+1):
    proton_counts.append([])

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
RefMult_TOFMatch1 = np.zeros((a, a))
RefMult_TOFMatch_bins1 = np.zeros((2, a))
RefMult_BetaEta = np.zeros((400, a))
RefMult_BetaEta_bins = np.zeros((2, a))
p_t = np.zeros((2, a))  # bins included in this one
phi = np.zeros((2, a))  # bins included in this one
dca = np.zeros((2, a))  # bins included in this one
eta = np.zeros((2, a))  # bins included in this one
nHitsFit_charge = np.zeros((2, b))  # bins included in this one
nHits_dEdX = np.zeros((2, c))  # bins included in this one
m_pq = np.zeros((a, a))
m_pq_bins = np.zeros((2, a))  # bins included in this one
beta_p = np.zeros((a, a))
beta_p_bins = np.zeros((2, a))  # bins included in this one
dEdX_pq = np.zeros((a, a))
dEdX_pq_bins = np.zeros((2, a))  # bins included in this one
ref_sum = np.zeros((2, a))  # bins included in this one
proton_sum = []
for i in range(a):
    proton_sum.append([])

# This is for the run ID (to identify runs to skip from QA).
Runs = []
# And it's nice to have a tally of the events you went over
# for the current analysis.
Events = 0

os.chdir(dataDirect)
r = len(os.listdir())
r_ = 1300  # For loop cutoff (to test on smaller batches).
count = 1
print("Working on file:")
for file in sorted(os.listdir()):
    # This cuts off the loop when testing.

    if count > 5:
        break

    # This is to cut off the beginning of the loop.
    """
    if count < r_:
        count += 1
        continue
    """
    # This is just to show how far along the script is.
    if count % 20 == 0:
        print(count, "of", r)
    # This is to make sure it's a ROOT file (it will error otherwise).
    if Path(file).suffix != '.root':
        continue
    # This is to omit all runs marked "bad."
    run_num = file[:-5]
    if run_num in bad_runs:
        print("Run", run_num, "skipped for being marked bad.")
        print("Bad, I tell you. Bad!!")
        r -= 1
        continue
    # Yu's cutoff for nSigmaProton and dE/dx calibration issues.
    if int(run_num) > 20118040:
        print("Over the threshold Yu had set for display (20118040).")
        r -= 1
        break

    data = up.open(file)
    try:
        data = data["PicoDst"]
    except ValueError:  # Skip empty picos.
        r -= 1
        print("Run number", run_num, "is empty.")  # Identifies the misbehaving file.
        continue
    except KeyError:  # Skip non empty picos that have no data.
        r -= 1
        print("Run number", run_num, "has no data.")  # Identifies the misbehaving file.
        continue
    except Exception as e:  # For any other issues that might pop up.
        print("Darn it!", e.__class__, "occurred in run", run_num)
        r -= 1
        continue

    # These are the event and track loops. It's written such that it
    # will ignore any iterations that are way "out of bounds." This shouldn't
    # be necessary, but it's here just in case. The stuff after "try" is
    # the "loop" proper (since we're in Python, it's not really a loop ...),
    # and the following "except" is for the really high outliers ("IndexError"
    # means the indicies don't match). This will toss all the data from that
    # run, not just the outlier that caused the issue, so this is only for
    # troubleshooting. If you did your job right, you'll never see the error.
    # If you see the error, something needs to be corrected.
    try:
        Runs.append(file[:-5])

        # Event level quantities and cuts.
        vX = ak.to_numpy(ak.flatten(data["Event"]["Event.mPrimaryVertexX"].array()))
        vY = ak.to_numpy(ak.flatten(data["Event"]["Event.mPrimaryVertexY"].array()))
        vZ = ak.to_numpy(ak.flatten(data["Event"]["Event.mPrimaryVertexZ"].array()))
        vR = np.power((np.power(vX, 2)+np.power(vY, 2)), (1/2))
        ZDCx = ak.to_numpy(ak.flatten(data["Event"]["Event.mZDCx"].array()))

        # Let's make some event cuts.
        event_cuts = ((vR <= 2.0) & (np.absolute(vZ) <= 30.0))
        vX, vY, vZ, vR = vX[event_cuts], vY[event_cuts], vZ[event_cuts], vR[event_cuts]

        RefMult3 = ak.to_numpy(ak.flatten(data["Event"]["Event.mRefMult3PosEast"].array() +
                                          data["Event"]["Event.mRefMult3PosWest"].array() +
                                          data["Event"]["Event.mRefMult3NegEast"].array() +
                                          data["Event"]["Event.mRefMult3NegWest"].array())[event_cuts])
        TofMult = ak.to_numpy(ak.flatten(data["Event"]["Event.mbTofTrayMultiplicity"].array())[event_cuts])
        TofMatch_ = ak.to_numpy(ak.flatten(data["Event"]["Event.mNBTOFMatch"].array())[event_cuts])

        pX = data["Track"]["Track.mGMomentumX"].array()[event_cuts]
        pY = data["Track"]["Track.mGMomentumY"].array()[event_cuts]
        pY = ak.where(pY == 0.0, 1e-10, pY)  # to avoid infinities
        pZ = data["Track"]["Track.mGMomentumZ"].array()[event_cuts]
        pT = np.power((np.power(pX, 2) + np.power(pY, 2)), (1 / 2))
        pG = np.power((np.power(pX, 2) + np.power(pY, 2) + np.power(pZ, 2)), (1 / 2))

        Eta = np.arcsinh(np.divide(pZ, np.sqrt(np.add(np.power(pX, 2), np.power(pY, 2)))))
        rapidity = fn.rapidity_awk(pZ)

        DcaX = data["Track"]["Track.mOriginX"].array()[event_cuts]-vX
        DcaY = data["Track"]["Track.mOriginY"].array()[event_cuts]-vY
        DcaZ = data["Track"]["Track.mOriginZ"].array()[event_cuts]-vZ
        Dca = np.power((np.power(DcaX, 2) + np.power(DcaY, 2) + np.power(DcaZ, 2)), (1/2))

        nHitsDedx = data["Track"]["Track.mNHitsDedx"].array()[event_cuts]
        nHitsFit = data["Track"]["Track.mNHitsFit"].array()[event_cuts]
        nHitsMax = data["Track"]["Track.mNHitsMax"].array()[event_cuts]
        nHitsMax = ak.where(nHitsMax == 0, 1e-10, nHitsMax)  # to avoid infinities
        dEdX = data["Track"]["Track.mDedx"].array()[event_cuts]
        nSigmaProton = data["Track"]["Track.mNSigmaProton"].array()[event_cuts]

        # We need to split our tracks into those which are Tof matched and those which aren't.
        # Track level quantities. Only Tof match for p > 0.8.
        TofPid = data["BTofPidTraits"]["BTofPidTraits.mTrackIndex"].array()[event_cuts]
        # Scaling for beta is a STAR thing; see StPicoBTofPidTraits.h.
        beta = data["BTofPidTraits"]["BTofPidTraits.mBTofBeta"].array()[event_cuts]/20000.0
        # Here are the Tof matched arrays.
        pT_, pG_, Eta_, rapidity_,\
            Dca_, nHitsDedx_, nHitsFit_,\
            nHitsMax_, dEdX_, nSigmaProton_, = \
            pT[TofPid],\
            pG[TofPid], Eta[TofPid],\
            rapidity[TofPid], Dca[TofPid],\
            nHitsDedx[TofPid], nHitsFit[TofPid],\
            nHitsMax[TofPid], dEdX[TofPid], \
            nSigmaProton[TofPid]

        # Now for some track level cuts.
        track_cuts = ((nHitsDedx > 5) & (np.absolute(nHitsFit) > 20) &
                      (np.divide(np.absolute(nHitsFit), nHitsMax) > 0.52) &
                      (Dca < 1.0) & (np.absolute(rapidity) <= 0.5) &
                      (pT >= 0.2) & (pG <= 10.0))
        pT, pG, Eta, rapidity,\
            Dca, nHitsDedx, nHitsFit,\
            nHitsMax, dEdX, nSigmaProton, = \
            pT[track_cuts],\
            pG[track_cuts], Eta[track_cuts],\
            rapidity[track_cuts], Dca[track_cuts],\
            nHitsDedx[track_cuts], nHitsFit[track_cuts],\
            nHitsMax[track_cuts], dEdX[track_cuts], \
            nSigmaProton[track_cuts]

        # And the same for our Tof matched tracks.
        track_cuts_ = ((nHitsDedx_ > 5) & (np.absolute(nHitsFit_) > 20) &
                       (np.divide(np.absolute(nHitsFit_), nHitsMax_) > 0.52) &
                       (Dca_ < 1.0) & (np.absolute(rapidity_) <= 0.5) &
                       (pT_ >= 0.2) & (pG_ <= 10.0))
        pT_, pG_, Eta_, rapidity_,\
            Dca_, nHitsDedx_, nHitsFit_,\
            nHitsMax_, dEdX_, nSigmaProton_, \
            beta = \
            pT_[track_cuts_],\
            pG_[track_cuts_], Eta_[track_cuts_],\
            rapidity_[track_cuts_], Dca_[track_cuts_],\
            nHitsDedx_[track_cuts_], nHitsFit_[track_cuts_],\
            nHitsMax_[track_cuts_], dEdX_[track_cuts_], \
            nSigmaProton_[track_cuts_], beta[track_cuts_]

        # Now for another event level cut (on RefMults). This is
        # for pileup and unknown nonconformity rejection.
        beta_eta1_match = ak.where((beta > 0.1) & (np.absolute(Eta_) < 1.0) &
                                   (np.absolute(Dca_ < 3.0) & (np.absolute(nHitsFit_) > 10)),
                                   1, 0)
        beta_eta1 = ak.sum(beta_eta1_match, axis=-1)
        TofMatch_match = ak.where((Eta < 0.5) & (Dca < 3) & (np.absolute(nHitsFit) > 10), 1, 0)
        TofMatch = ak.sum(TofMatch_match, axis=-1)

        event_cuts2 = ((TofMult >= (1.352*RefMult3 - 54.08)) & (TofMult <= (2.536*RefMult3 + 200)) &
                       (TofMatch >= (0.239*RefMult3 - 14.34)) & (beta_eta1 >= (0.447*RefMult3 - 17.88)))

        pT, pG, Eta, rapidity, Dca, nHitsDedx, nHitsFit, nHitsMax, dEdX, nSigmaProton,\
            RefMult3, TofMult, TofMatch, beta_eta1 = \
            pT[event_cuts2],\
            pG[event_cuts2], Eta[event_cuts2], rapidity[event_cuts2], Dca[event_cuts2],\
            nHitsDedx[event_cuts2], nHitsFit[event_cuts2], nHitsMax[event_cuts2], \
            dEdX[event_cuts2], nSigmaProton[event_cuts2], RefMult3[event_cuts2], TofMult[event_cuts2],\
            TofMatch[event_cuts2], beta_eta1[event_cuts2]
        pT_, pG_, Eta_, rapidity_, Dca_, nHitsDedx_, nHitsFit_, nHitsMax_, beta, dEdX_, nSigmaProton_,\
            RefMult3_, TofMult_, TofMatch_ = \
            pT_[event_cuts2], pG_[event_cuts2], Eta_[event_cuts2], rapidity_[event_cuts2], Dca_[event_cuts2],\
            nHitsDedx_[event_cuts2], nHitsFit_[event_cuts2], nHitsMax_[event_cuts2], beta[event_cuts2],\
            dEdX_[event_cuts2], nSigmaProton_[event_cuts2], RefMult3_[event_cuts2], TofMult_[event_cuts2],\
            TofMatch_[event_cuts2]
        Events += len(pT)  # Final event count
        # charge = ak.where(nHitsFit >= 0, 1, -1)

        # Now that we have all our quantities, let's find some protons!

        # Calibration of nSigmaProton for 0.0 < |p| < 0.8 (assumed 0 otherwise)
        # First, we'll separate it into discrete groupings of |p|.
        sig_length = 19
        nSigmaProton_p = []
        pT_nS = np.asarray(ak.flatten(pT))
        nS_nS = np.asarray(ak.flatten(nSigmaProton))
        nSigmaProton_p.append(nS_nS[(pT_nS <= 0.2)])
        for k in range(2, sig_length+1):
            nSigmaProton_p.append(nS_nS[((pT_nS > 0.1*k) & (pT_nS <= 0.1*(k+1)))])
        nSigmaProton_p = np.asarray(nSigmaProton_p)

        # Now to find the peak of the proton distribution. I'm going to try smoothing the
        # distributions, then finding the inflection points via a second order derivative.
        sig_means = []
        p_count = 0
        # Turned off for now; it can be turned back on if we use all runs and not just
        # the ones before Yu's hard cutoff.
        """
        for dist in nSigmaProton_p:
            counter, bins = np.histogram(dist, range=(-10000, 10000), bins=200)
            sgfProton_3 = sgf(counter, 45, 2)
            sgfProton_3_2 = sgf(sgfProton_3, 45, 2, deriv=2)
            infls = bins[:-1][np.where(np.diff(np.sign(sgfProton_3_2)))[0]]
            sig_mean = 0
            if infls.size >= 2:
                infls_bounds = np.sort(np.absolute(infls))
                first = infls[np.where(np.absolute(infls) == infls_bounds[0])[0][0]]
                second = infls[np.where(np.absolute(infls) == infls_bounds[1])[0][0]]
                if first > second:
                    sig_mean = first-(first-second)/2
                else:
                    sig_mean = second-(second-first)/2
            if p_count >= 10:
                sig_mean = 0
            sig_means.append(sig_mean)
            # The below is to check things; turned off if running over lots of files.
            # plt.plot(bins[:-1], counter, c="blue", lw=2, label="Raw")
            # plt.plot(bins[:-1], sgfProton_3, c="red", lw=1, label="Smoothed")
            # plt.plot(bins[:-1], sgfProton_3_2, c="green", label="2nd derivative")
            # for k, infl in enumerate(infls, 1):
            #    plt.axvline(x=infl, color='k', label=f'Inflection Point {k}')
            # plt.axvline(x=sig_mean, c="pink", label="nSigmaMean")
            # p_title = r'$p_T$ <= ' + str(0.1*(p_count+2))
            # plt.title(p_title)
            # plt.legend()
            # plt.show()
            p_count += 1
        sig_means = np.asarray(sig_means)

        # Now to modify nSigmaProton to be the difference between the values and
        # the found means.
        nSigmaProton = ak.where(pT <= 0.2, nSigmaProton-sig_means[0], nSigmaProton)
        for k in range(1, len(sig_means)):
            nSigmaProton = ak.where((pT > 0.1*(k+1)) & (pT <= 0.1*(k+2)), nSigmaProton-sig_means[k], nSigmaProton)
        """
        # Let's do a mass cut.
        p_squared = np.power(pG_, 2)
        print("1")
        b_squared = np.power(beta, 2)
        print("2")
        b_squared = ak.where(b_squared == 0.0, 1e-10, b_squared)  # to avoid infinities
        print("3")
        g_squared = (1 - np.power(beta, 2))
        print("4")
        m_squared = np.divide(np.multiply(p_squared, g_squared), b_squared)
        print("5")
        # All our proton selection criteria.
        charge = ak.where((pT > 0.4) & (pT < 2.0) & (abs(nSigmaProton) <= 2000), charge, 0)
        charge_1 = ak.where((pT*abs(charge) < 0.8) & (pG*abs(charge) <= 1.0), charge, 0)
        charge_2 = ak.where((pT*abs(charge) >= 0.8) & (pG*abs(charge) <= 3.0) &
                            (m_squared*abs(charge) >= 0.6) & (m_squared*abs(charge) <= 1.2),
                            charge, 0)
        charge = charge_1 + charge_2

        mass_cut = (charge != 0)
        pT, pG, Eta, rapidity, Dca, nHitsDedx, nHitsFit, nHitsMax, beta,\
            dEdX, nSigmaProton, charge, m_squared =\
            pT[mass_cut], pG[mass_cut], Eta[mass_cut],\
            rapidity[mass_cut], Dca[mass_cut], nHitsDedx[mass_cut], nHitsFit[mass_cut],\
            nHitsMax[mass_cut], beta[mass_cut], dEdX[mass_cut], nSigmaProton[mass_cut],\
            charge[mass_cut], m_squared[mass_cut]

        # And here are the protons (we hope). I need to make a similar set for
        # protons and antiprotons separately.
        protons = ak.sum(charge, axis=-1)
        protons = ak.to_numpy(protons)
        for i in range(a):
            proton_sum[i].append(protons[RefMult3 == i])
        Ave_proton[0].append(np.mean(protons))
        Ave_proton[1].append(np.divide(np.std(protons), len(protons)))

        # *** Turns out the below is garbage. Fixed, but kept in until I finalise the code. ***

        # Now we separate the proton counts by RefMult bin.
        # This code need to get a LOT tighter (but it's pretty quick as is).
        proton_counts[0] = np.hstack((proton_counts[0], np.asarray(protons[(RefMult3 >= RefCuts[8])])))
        proton_counts[9] = np.hstack((proton_counts[9], np.asarray(protons[(RefMult3 < RefCuts[0])])))
        for j in range(1, 9):
            x = int(9-j)
            proton_counts[j] = np.hstack((proton_counts[j],
                                          np.asarray((protons[((RefMult3 < RefCuts[x]) &
                                                               (RefMult3 >= RefCuts[x - 1]))]))))

        # *** End of the garbage. ***

        # Now to convert our awkward arrays to NumPy for processing.
        pT = ak.to_numpy(ak.flatten(pT))
        pG = ak.to_numpy(ak.flatten(pG))
        dEdX = ak.to_numpy(ak.flatten(dEdX))
        nHitsFit = ak.to_numpy(ak.flatten(nHitsFit))
        nHitsDedx = ak.to_numpy(ak.flatten(nHitsDedx))
        Dca = ak.to_numpy(ak.flatten(Dca))
        Eta = ak.to_numpy(ak.flatten(Eta))
        beta_eta1 = ak.to_numpy(beta_eta1)
        m_squared = ak.to_numpy(ak.flatten(m_squared))
        charge = ak.to_numpy(ak.flatten(charge))
        beta = ak.where(beta == 0.0, 1e-10, beta)  # To avoid infinities
        beta = np.power(beta, -1)  # For graphing
        beta = ak.to_numpy(ak.flatten(beta))
        TofMatch = ak.to_numpy(TofMatch)
        TofMatch_ = ak.to_numpy(TofMatch_)

        # Now to fill our histograms.

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

        # transverse vertex position
        counter, binsX, binsY = np.histogram2d(vY, vX, bins=a,
                                               range=((-10, 10), (-10, 10)))
        v_r += counter
        v_r_bins = (binsX, binsY)

        # ToFMult vs RefMult
        counter, binsX, binsY = np.histogram2d(TofMult, RefMult3, bins=(1700, a),
                                               range=((0, 1700), (0, 1000)))
        RefMult_TOFMult += counter
        RefMult_TOFMult_bins = np.array((binsX, binsY), dtype=object)

        # nToFMatch vs RefMult3
        counter, binsX, binsY = np.histogram2d(TofMatch, RefMult3, bins=a,
                                               range=((0, 500), (0, 1000)))
        RefMult_TOFMatch += counter
        RefMult_TOFMatch_bins = (binsX, binsY)

        # nToFMatch vs RefMult3 (check to see if we don't have to calculate it)
        counter, binsX, binsY = np.histogram2d(TofMatch_, RefMult3, bins=a,
                                               range=((0, 500), (0, 1000)))
        RefMult_TOFMatch1 += counter
        RefMult_TOFMatch_bins1 = (binsX, binsY)

        # RefMult vs beta_eta1
        counter, binsX, binsY = np.histogram2d(beta_eta1, RefMult3,
                                               bins=(400, a), range=((0, 400), (0, 1000)))
        RefMult_BetaEta += counter
        RefMult_BetaEta_bins = np.array((binsX, binsY), dtype=object)

        # And that's the end of the loop.
        count += 1
    except Exception as e:  # For any issues that might pop up.
        print(e.__class__, "occurred in the main loop.")
        count += 1
        continue

print("All files analysed.")
print("Total events:", Events)

# Let's save where we want to save.
os.chdir(finalDirect)

# Proton distributions per RefMult3:
proton_sum = np.asarray(proton_sum)
np.save("proton_sum.npy", proton_sum)

# First to save the average values per run.
Ave_proton = np.asarray(Ave_proton)
np.save("Ave_protons.npy", Ave_proton)

# *** Oh, look: more garbage! ***
# Then save all the proton values.
proton_counts = np.asarray(proton_counts)
np.save("protons.npy", proton_counts)
# *** End of the garbage. ***

# Now to save the list of runs for QA.
Runs = np.asarray(Runs)
np.save("run_list.npy", Runs)

# Now to save our histograms for reconstruction.
np.save("vR.npy", v_r)
np.save("vR_bins.npy", v_r_bins)
np.save("refmult_tofmult.npy", RefMult_TOFMult)
np.save("refmult_tofmult_bins.npy", RefMult_TOFMult_bins)
np.save("refmult_tofmatch.npy", RefMult_TOFMatch)
np.save("refmult_tofmatch_bins.npy", RefMult_TOFMatch_bins)
np.save("refmult_tofmatch1.npy", RefMult_TOFMatch1)
np.save("refmult_tofmatch_bins1.npy", RefMult_TOFMatch_bins1)
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
