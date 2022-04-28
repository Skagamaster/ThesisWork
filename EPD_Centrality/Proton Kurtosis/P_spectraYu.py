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

# This is where our data lives and where we want to store the
# output arrays.
dataDirect = r"E:\2019Picos\14p5GeV\Runs"
# For the initial cuts
saveDirect = r"C:\Users\dansk\Documents\Thesis\AfterCuts_Initial"
#---------------------------------------------------------------#
# This bit of code is to exclude runs that have been determined
# to be outside of parameters. Turn this "off" if you're running
# over data to determine which runs are bad.
os.chdir(saveDirect)
badRuns = np.load("badRuns.npy", allow_pickle=True)
# Here are the runs Yu determined to be bad.
yu_badRuns = np.loadtxt(
    r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\yubad.txt")
# And this is for the RefMult3 cuts (which you won't have if
# you haven't yet determined them, so turn this off on initial
# runs of the code).
# RefCuts = np.load("RefMult_for_cent.npy", allow_pickle=True)
# Here's the RefMult3 cuts from Yu (as an override):
RefCuts = np.asarray((10, 21, 41, 72, 118, 182, 270, 392, 472))
# For the proton extrapolation after initial analysis (turn off
# for initial analysis)
saveDirect = r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu"
# Something to hold our proton numbers:
netProtonCount = []
for i in range(int(len(RefCuts))+1):
    netProtonCount.append([])
# Momentum numbers from Yu:
yu_runs = np.loadtxt(
    r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\runs.txt", delimiter=',')
yu_nsigma_values = np.loadtxt(
    r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\nsigmaVals.txt", delimiter=',')
#---------------------------------------------------------------#

# Arrays to hold our histogram data. These are essentially like ROOT
# histograms, but we'll fill them with the numpy histogram info. We
# need to have a separate array for the bins (this should go to pandas
# in the future as it's easier to keep track, but alas we live in the
# present).
a = 1000
b = 100
dEdXpQ = np.zeros((a, a))
dEdXpQbins = np.zeros((2, a))
dEdXpQproton = np.zeros((a, a))
dEdXpQbinsproton = np.zeros((2, a))
vR = np.zeros((a, a))
vRbins = np.zeros((2, a))
vZ = np.zeros((2, a))  # bins included in this one
betaP = np.zeros((a, a))
betaPbins = np.zeros((2, a))
RefMult = np.zeros((2, a))  # bins included in this one
RingRef = np.zeros((32, a))
RingRefbins = np.zeros((32, a))
nSigmaProtonPico = np.zeros((2, a))  # bins included in this one

# The following are all for the average value for an event.
# The second row is for the error.
AveVr = [[], []]
AveVz = [[], []]
AveEta = [[], []]
AvePhi = [[], []]
AvePt = [[], []]
AveDca = [[], []]
AveZdcX = [[], []]
AveRefmult3 = [[], []]

# This is for the run ID (to identify runs to skip from QA).
Runs = []
Events = 0

# This is to get the days for calibration.
caliDays = np.asarray(("94", "105", "110", "113", "114", "123", "138", "139"))

# This is for the nSigmaProton mean values.
nSigmaBins = []

os.chdir(dataDirect)
r = len(os.listdir())
count = 1
print("Working on file:")
for file in os.listdir():
    run_num = int(file[:-5])
    # This is just to show how far along the script is.
    if (count) % 100 == 0:
        print(count, "of", r)
    # This is to make sure it's a ROOT file (it will error otherwise).
    if Path(file).suffix != '.root':
        continue
    # Use this to cut off the loop for testing.
    if count > 3:
        break
    if file[:-5] in badRuns:
        count += 1
        print("Skipped run", file[:-5], "due to being marked bad.")
        continue
    if run_num in yu_badRuns:
        count += 1
        print("Skipped run", file[:-5], "due to being marked bad by Yu.")
        continue
    data = up.open(file)
    try:
        data = data["PicoDst"]
    except ValueError:  # Skip empty picos.
        count += 1
        print("ValueError at", file)  # Identifies the misbehaving file.
        continue
    except KeyError:  # Skip non empty picos that have no data.
        count += 1
        print("KeyError at", file)  # Identifies the misbehaving file.
        continue
    count += 1

    # This is to get calibrated EPD nMIPs. These are 744 tiles, 0-743.
    fileLoc = r'D:\14GeV\ChiFit\Nmip_Day_'
    dayFit = file[2:5]
    fileDay = np.where(caliDays.astype(int) < int(dayFit))[0][-1]
    nFits = np.loadtxt(fileLoc + caliDays[fileDay] + ".txt")[:, 4]

    # This will skip any values that are way out of bounds. It shouldn't
    # be necessary, but it's here just in case. The stuff after "try" is
    # the loop proper, and the following "except" is for the really
    # high outliers ("IndexError" means the indicies don't match).
    # This will toss all the data from that run, not just the outlier that
    # caused the issue, so this is only for troubleshooting. If you did
    # your job right, you'll never see the error.
    try:
        Runs.append(file[:-5])
        vX = np.hstack(np.asarray(
            data["Event"]["Event.mPrimaryVertexX"].array()))
        vY = np.hstack(np.asarray(
            data["Event"]["Event.mPrimaryVertexY"].array()))
        vZa = np.hstack(np.asarray(
            data["Event"]["Event.mPrimaryVertexZ"].array()))
        vRa = np.power((np.power(vX, 2)+np.power(vY, 2)), (1/2))
        # Vertex cuts.
        eventCuts = fn.vertCuts(vRa, vZa, 2.0, 30.0)
        # Now to cut the events.
        vX = vX[eventCuts]
        vY = vY[eventCuts]
        vZa = vZa[eventCuts]
        vRa = vRa[eventCuts]
        events = int(len(eventCuts))
        Events += events

        # Event quantities (after cuts).
        RefMult3 = np.hstack(np.asarray(data["Event"]["Event.mRefMult3PosEast"].array()) +
                             np.asarray(data["Event"]["Event.mRefMult3PosWest"].array()) +
                             np.asarray(data["Event"]["Event.mRefMult3NegEast"].array()) +
                             np.asarray(data["Event"]["Event.mRefMult3NegWest"].array()))[
                                 eventCuts]

        # EPD ring information
        epdID = np.asarray(data["EpdHit"]["EpdHit.mId"].array())[eventCuts]
        nMIP = np.asarray(data["EpdHit"]["EpdHit.mnMIP"].array())[eventCuts]
        RingRefTaco = fn.epdRing(epdID)
        RingRefPico = []
        for q in range(32):
            RingRefPico.append([])
        RingRefTaco = np.asarray(RingRefTaco)

        # Track quantities (QA cuts after ToF matching).
        DcaX = np.asarray(data["Track"]["Track.mOriginX"].array())[
            eventCuts]-vX
        DcaY = np.asarray(data["Track"]["Track.mOriginY"].array())[
            eventCuts]-vY
        DcaZ = np.asarray(data["Track"]["Track.mOriginZ"].array())[
            eventCuts]-vZa
        Dca = np.power(
            (np.power(DcaX, 2) + np.power(DcaY, 2) + np.power(DcaZ, 2)), (1/2))
        nHitsDedx = np.asarray(data["Track"]["Track.mNHitsDedx"].array())[
            eventCuts]
        nHitsFit = np.asarray(data["Track"]["Track.mNHitsFit"].array())[
            eventCuts]
        charge = fn.chargeList(nHitsFit)
        nHitsMax = np.asarray(data["Track"]["Track.mNHitsMax"].array())[
            eventCuts]
        # Scaling here is a STAR thing; see StPicoBTofPidTraits.h.
        beta = np.asarray((data["BTofPidTraits"]
                           ["BTofPidTraits.mBTofBeta"].array())/20000.0)[eventCuts]
        TofPid = np.asarray(data["BTofPidTraits"]
                            ["BTofPidTraits.mTrackIndex"].array())[eventCuts]
        pX = np.asarray(data["Track"]["Track.mGMomentumX"].array())[eventCuts]
        pY = np.asarray(data["Track"]["Track.mGMomentumY"].array())[eventCuts]
        pZ = np.asarray(data["Track"]["Track.mGMomentumZ"].array())[eventCuts]
        pG = np.power(
            (np.power(pX, 2) + np.power(pY, 2) + np.power(pZ, 2)), (1/2))
        pG = np.multiply(pG, charge)
        pT = np.power((np.power(pX, 2) +
                       np.power(pY, 2)), (1/2))
        dEdX = np.asarray(data["Track"]["Track.mDedx"].array())[eventCuts]
        nSigmaProton = np.asarray(
            data["Track"]["Track.mNSigmaProton"].array())[eventCuts]
        dEdXproton = np.copy(dEdX)
        pGproton = np.copy(pG)
        betaproton = np.copy(beta)
        nSigmaProtonNoCut = np.copy(nSigmaProton)

        # Get the nSigmaProton mean positions from Yu's code.
        # These will be in the following p bins:
        # 0.1 increments from 0.1-1.3, then 0 definite above 1.3.
        # This is because Kaon intrusion into the nSigmaProton
        # range of the protons above 1.3.
        if run_num not in yu_runs:
            continue
        selector = np.hstack(np.asarray(np.where(yu_runs == run_num)))[0]
        nspMean1 = yu_nsigma_values[selector]
        pRanges = np.linspace(0, 1.2, 13)
        # ToF matching all track arrays and net proton multiplicity.
        for i in range(int(len(eventCuts))):
            try:
                index = TofPid[i]  # ToF match cut.
                pZ[i], pG[i], pGproton[i], pT[i], charge[i], nHitsFit[i], nHitsDedx[i], \
                    nHitsMax[i], Dca[i], dEdX[i], dEdXproton[i], nSigmaProton[i] = fn.indexer(
                    index, pZ[i], pG[i], pGproton[i], pT[i], charge[i], nHitsFit[i],
                    nHitsDedx[i], nHitsMax[i], Dca[i], dEdX[i], dEdXproton[i], nSigmaProton[i])
                # Now we'll cut our tracks.
                trackCuts = fn.trackCutsTest(
                    Dca[i], nHitsDedx[i], nHitsFit[i], nHitsMax[i], beta[i], pG[i],
                    dEdX[i], 1.0, 5, 20)
                beta[i], betaproton[i], pZ[i], pG[i], pGproton[i], pT[i], charge[i], \
                    nHitsFit[i], nHitsDedx[i], Dca[i], dEdX[i], dEdXproton[i], \
                    nSigmaProton[i] = \
                    fn.indexer(trackCuts, beta[i], betaproton[i],
                               pZ[i], pG[i], pGproton[i], pT[
                        i], charge[i], nHitsFit[i],
                    nHitsDedx[i], Dca[i], dEdX[i], dEdXproton[i], nSigmaProton[i])

                # Proton selection
                '''
                testIndex = np.where((np.absolute(pGproton[i]) < 1) & (
                    np.absolute(pGproton[i]) > 0.24))
                beta[i], pG[i], betaproton[i], pGproton[i], pT[i], charge[i], nHitsFit[i], nHitsDedx[i], \
                    Dca[i], dEdXproton[i], nSigmaProton[i] = \
                    fn.indexer(testIndex, beta[i], pG[i], betaproton[i], pGproton[i],
                               pT[i], charge[i], nHitsFit[i], nHitsDedx[i],
                               Dca[i], dEdXproton[i], nSigmaProton[i])
                '''
                mSquared = np.divide(np.multiply(
                    np.power(pG[i], 2), (1-np.power(beta[i], 2))), np.power(beta[i], 2))
                mIndex = np.where((mSquared >= 0.6) & (mSquared <= 1.2))

                if np.asarray(mIndex).size == 0:
                    continue
                betaproton[i], pGproton[i], pT[i], charge[i], nHitsFit[i], nHitsDedx[i], \
                    Dca[i], dEdXproton[i], nSigmaProton[i] = \
                    fn.indexer(mIndex, betaproton[i], pGproton[i],
                               pT[i], charge[i], nHitsFit[i], nHitsDedx[i],
                               Dca[i], dEdXproton[i], nSigmaProton[i])
                lowPro = fn.protonCutsBeta(pT[i], pGproton[i], nSigmaProton[i], nspMean1,
                                           pG[i], pZ[i], pRanges, pTlow=0.4, pThigh=0.8, pGmax=1.0,
                                           rapMax=0.5, i=i)
                highPro = fn.protonCutsBeta(pT[i], pGproton[i], nSigmaProton[i], nspMean1,
                                            pG[i], pZ[i], pRanges, pTlow=0.8, pThigh=2.0, pGmax=3.0,
                                            rapMax=0.5, i=i)
                # This is to see how we're doing on proton selection.
                pGprotonLow, dEdXprotonLow = fn.indexer(
                    lowPro, pGproton[i], dEdXproton[i])
                pGprotonHigh, dEdXprotonHigh = fn.indexer(
                    highPro, pGproton[i], dEdXproton[i])
                pGproton[i] = np.hstack((pGprotonLow, pGprotonHigh))
                dEdXproton[i] = np.hstack((dEdXprotonLow, dEdXprotonHigh))
                # Now to count our (net) protons.
                protonsLow = len(np.hstack(np.asarray(
                    np.where(charge[i][lowPro] >= 0))))
                antiprotonsLow = len(
                    np.hstack(np.asarray(np.where(charge[i][lowPro] < 0))))
                protonsHigh = len(np.hstack(np.asarray(
                    np.where(charge[i][highPro] >= 0))))
                antiprotonsHigh = len(
                    np.hstack(np.asarray(np.where(charge[i][highPro] < 0))))
                netProton = protonsLow - antiprotonsLow + protonsHigh - antiprotonsHigh
                # Sort the (net) proton multiplicities by RefMult3.
                if RefMult3[i] > RefCuts[int(len(RefCuts))-1]:
                    netProtonCount[int(len(RefCuts))].append(netProton)
                if RefMult3[i] <= RefCuts[0]:
                    netProtonCount[0].append(netProton)
                for l in range(int(len(RefCuts))-1):
                    if (RefMult3[i] > RefCuts[l]) & (RefMult3[i] <= RefCuts[l+1]):
                        netProtonCount[l+1].append(netProton)
                # Fill the EPD rings (for EPD centrality)
                tiles = fn.EPDTile(epdID[i])
                calVal = nFits[tiles]
                # nMIP calibration and truncation
                fitMIP = nMIP[i]/calVal
                fitMIP[fitMIP > 3.0] = 3.0
                fitMIP[fitMIP < 0.2] = 0.0
                for l in range(32):
                    try:
                        index = np.where(RingRefTaco[i] == l)
                        RingRefPico[l].append(np.sum(fitMIP[index]))
                    except IndexError:
                        print("Index error in EPD loop at", i, l)
            except IndexError:  # For if there are no ToF matches in an event.
                print("Index error in TOF/EPD loop at", i)
                continue
        charge = np.hstack(charge)
        pT = np.hstack(pT)
        pG = np.hstack(pG)
        dEdX = np.hstack(dEdX)
        pGproton = np.hstack(pGproton)
        dEdXproton = np.hstack(dEdXproton)
        nHitsFit = np.hstack(nHitsFit)
        nHitsDedx = np.hstack(nHitsDedx)
        Dca = np.hstack(Dca)
        beta = np.power(beta, -1)  # For graphing
        beta = np.hstack(beta)
        beta[beta > 100] = 100  # To avoid infinities
        betaproton = np.power(betaproton, -1)  # For graphing
        betaproton = np.hstack(betaproton)
        betaproton[betaproton > 100] = 100  # To avoid infinities
        RingRefPico = np.asarray(RingRefPico)
        nSigmaProton = np.hstack(nSigmaProton)
        nSigmaProtonNoCut = np.hstack(nSigmaProtonNoCut)

        # Now to fill our histograms.
        #-----de/dx vs pc-----#
        counter, binsX, binsY = np.histogram2d(
            dEdX, pG, bins=a, range=((0, 20), (-5, 5)))
        dEdXpQ += counter
        dEdXpQbins = (binsX, binsY)
        #-----de/dx vs pc after proton selection-----#
        counter, binsX, binsY = np.histogram2d(
            dEdXproton, pGproton, bins=a, range=((0, 20), (-5, 5)))
        dEdXpQproton += counter
        dEdXpQbinsproton = (binsX, binsY)
        #-----transverse vertex position-----#
        counter, binsX, binsY = np.histogram2d(
            vY, vX, bins=a, range=((-10, 10), (-10, 10)))
        vR += counter
        vRbins = (binsX, binsY)
        #-----1/beta vs p-----#
        counter, binsX, binsY = np.histogram2d(
            beta, abs(pG), bins=a)
        betaP += counter
        betaPbins = (binsX[:-1], binsY[:-1])
        #-----z vertex position-----#
        counter, bins = np.histogram(vZa, bins=a, range=(-200, 200))
        vZ[0] += counter
        vZ[1] = bins[:-1]
        #-----RefMult3-----#
        counter, bins = np.histogram(RefMult3, bins=a, range=(0, 1000))
        RefMult[0] += counter
        RefMult[1] = bins[:-1]
        for q in range(32):
            counter, bins = np.histogram(RingRefPico[q], bins=a, range=(0, 20))
            RingRef[q] += counter
            RingRefbins[q] = bins[:-1]
        # Averages for Run cuts in later analyis.
        # arr[0] is the mean, arr[1] is MSE.
        AveVz[0].append(np.mean(vZa))
        AveVz[1].append(np.std(vZa)/np.sqrt(events))
        AvePt[0].append(np.mean(pT))
        AvePt[1].append(np.std(pT)/np.sqrt(events))
        AveVr[0].append(np.mean(vRa))
        AveVr[1].append(np.std(vRa)/np.sqrt(events))
        AveDca[0].append(np.mean(Dca))
        AveDca[1].append(np.std(Dca)/np.sqrt(events))
        AveRefmult3[0].append(np.mean(RefMult3))
        AveRefmult3[1].append(np.std(RefMult3)/np.sqrt(events))
    except IndexError:
        # Identifies the misbehaving file.
        print("IndexError in main loop at", file)
        continue

# Let's save where we want to save.
os.chdir(saveDirect)

AveVz = np.asarray(AveVz)
AvePt = np.asarray(AvePt)
AveVr = np.asarray(AveVr)
Runs = np.asarray(Runs)
AveDca = np.asarray(AveDca)
AveRefmult3 = np.asarray(AveRefmult3)
netProtonCount = np.asarray(netProtonCount)

print("All files analysed.")
print("Total events:", Events)

np.save("netProtonCount.npy", netProtonCount)
np.save("RingRef.npy", RingRef)
np.save("dEdXpG.npy", dEdXpQ)
np.save("dEdXpGbins.npy", dEdXpQbins)
np.save("dEdXpGprotons.npy", dEdXpQproton)
np.save("dEdXpGbinsprotons", dEdXpQbinsproton)
np.save("vR.npy", vR)
np.save("vRbins.npy", vRbins)
np.save("betaP.npy", betaP)
np.save("betaPbins.npy", betaPbins)
np.save("RefMult.npy", RefMult)
np.save("RingRef.npy", RingRef)
np.save("vZ.npy", vZ)

# Pandas df of the averages.
columns = ["vZ", "vZerr", "pT", "pTerr", "vR", "vRerr",
           "Dca", "Dcaerr", "RefMult3", "RefMult3err"]
arr = np.asarray((AveVz[0], AveVz[1], AvePt[0], AvePt[1], AveVr[0], AveVr[1],
                  AveDca[0], AveDca[1], AveRefmult3[0], AveRefmult3[1]))
#df = pd.DataFrame(arr.T, index=Runs, columns=columns)
# df.to_pickle("averages.pkl")

'''
cmap = "icefire_r"
plt.pcolormesh(dEdXpQ, norm=colors.LogNorm(), cmap=cmap)
x = np.linspace(0, a, 11)
xp = np.linspace(-5, 5, 11)
xs = xp.astype(str)
y = np.linspace(0, a, 11)
yp = np.linspace(0, 20, 11)
ys = yp.astype(str)
plt.xticks(x, xs)
plt.yticks(y, ys)
plt.xlabel(r"$p_G$ ($\frac{GeV}{c}$)", fontsize=20)
plt.ylabel(r'$\frac{dE}{dX}$ ($\frac{KeV}{cm}$)', fontsize=20)
plt.title(r"Global Momentum vs Energy Loss", fontsize=30)
plt.colorbar()
plt.show()
plt.close()

plt.pcolormesh(vR, norm=colors.LogNorm(), cmap=cmap)
plt.show()
plt.close()

plt.pcolormesh(betaP, norm=colors.LogNorm(), cmap=cmap)
plt.show()
plt.close()

plt.plot(vZ)
plt.show()
plt.close()
'''
