import uproot as up
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
import seaborn as sns
import functions as fn

# This is where our data lives and where we want to store the
# output arrays.
dataDirect = r"E:\2019Picos\14p5GeV\Runs"
# For the initial cuts
saveDirect = r"C:\Users\dansk\Documents\Thesis\AfterCuts_Initial"

#################################################################
# This bit of code is to exclude runs that have been determined
# to be outside of parameters. Turn this "off" if you're running
# over data to determine which runs are bad.
os.chdir(saveDirect)
badRuns = np.load("badRuns.npy", allow_pickle=True)
# And this is for the RefMult3 cuts (which you won't have if
# you haven't yet determined them, so turn this off on initial
# runs of the code).
RefCuts = np.load("RefMult_for_cent.npy", allow_pickle=True)
# For the proton extrapolation after initial analysis (turn off
# for initial analysis)
saveDirect = r"C:\Users\dansk\Documents\Thesis\Protons"
# Something to hold our proton numbers:
netProtonCount = []
for i in range(int(len(RefCuts))+1):
    netProtonCount.append([])
#################################################################

# Arrays to hold our data. These are essentially like ROOT
# histograms (we'll set the upper limit later). The array
# simply sets the dimensions; the precision is set when you
# fill them. It's a bit clunkier than ROOT, oddly (as I find
# ROOT terribly clunky compared to Python), but it gets the
# job done.
a = 1000
b = 100  # Dimensionality for our "histograms."
dEdXpQ = np.zeros((a, a))
dEdXpQPRIME1 = np.zeros((a, a))
dEdXpQPRIME2 = np.zeros(a)
dEdXpQPRIME3 = np.zeros(a)
#dEdXpQPRIME = np.asarray((dEdXpQPRIME1, dEdXpQPRIME2, dEdXpQPRIME3))
vZ = np.zeros(a)
betaP = np.zeros((a, a))
RefMult = np.zeros(a)
RingRef = np.zeros((32, b))
nSigmaProtonPico = np.zeros((2, a))

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

os.chdir(dataDirect)
r = len(os.listdir())
count = 1
print("Working on file:")
for file in os.listdir():
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
        eventCuts = fn.vertCuts(vRa, vZa, 2.0, 70.0)
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

        # Track quantities for track QA cuts.
        DcaX = np.hstack(np.asarray(
            data["Track"]["Track.mOriginX"].array())[eventCuts]-vX)
        DcaY = np.hstack(np.asarray(
            data["Track"]["Track.mOriginY"].array())[eventCuts]-vY)
        DcaZ = np.hstack(np.asarray(
            data["Track"]["Track.mOriginZ"].array())[eventCuts]-vZa)
        Dca = np.power(
            (np.power(DcaX, 2) + np.power(DcaY, 2) + np.power(DcaZ, 2)), (1/2))
        nHitsDedx = np.hstack(np.asarray(data["Track"]["Track.mNHitsDedx"].array())[
            eventCuts])
        nHitsFit = np.asarray(data["Track"]["Track.mNHitsFit"].array())[
            eventCuts]
        #charge = fn.chargeList(nHitsFit)
        #nHitsFit = np.hstack(nHitsFit)
        # nHitsFitFull = np.asarray(data["Track"]["Track.mNHitsFit"].array())[
        #    eventCuts]

        # Some other track quantities we'll not cut on, yet. I need to redo all this!!!

        # Scaling here is a STAR thing; see StPicoBTofPidTraits.h.
        beta = np.asarray((data["BTofPidTraits"]
                           ["BTofPidTraits.mBTofBeta"].array())/20000.0)[eventCuts]
        beta = np.power(beta, -1)  # For graphing
        TofPid = np.asarray(data["BTofPidTraits"]
                            ["BTofPidTraits.mTrackIndex"].array())[eventCuts]
        pX = np.asarray(data["Track"]["Track.mGMomentumX"].array())[eventCuts]
        pY = np.asarray(data["Track"]["Track.mGMomentumY"].array())[eventCuts]
        pZ = np.asarray(data["Track"]["Track.mGMomentumZ"].array())[eventCuts]
        pGa = np.power(
            (np.power(pX, 2) + np.power(pY, 2) + np.power(pZ, 2)), (1/2))
        pGb = np.power(
            (np.power(pX, 2) + np.power(pY, 2) + np.power(pZ, 2)), (1/2))
        pT = np.power((np.power(pX, 2) +
                       np.power(pY, 2)), (1/2))
        pGsmall = []
        for i in range(int(len(eventCuts))):
            try:
                index = TofPid[i]
                pGb[i] = pGb[i][index]
                pT[i] = pT[i][index]
            except IndexError:  # For if there are no TOF matches in an event.
                print("Index error in TOF/EPD loop at", i)
                break
        for i in range(int(len(eventCuts))):
            try:
                index = TofPid[i]
                nHitsFit[i] = nHitsFit[i][index]
                nHitsDedx[i] = nHitsDedx[i][index]
                Dca[i] = Dca[i][index]
            except IndexError:  # For if there are no TOF matches in an event.
                print("Index error in TOF/EPD loop at", i)
                break
        break

        # Now for the track cuts.
        trackCuts = fn.trackCuts(Dca, nHitsDedx, nHitsFit, 1.0, 5, 20)
        # And cutting our existing tracks.
        Dca = Dca[trackCuts]
        nHitsDedx = nHitsDedx[trackCuts]
        nHitsFit = nHitsFit[trackCuts]

        dEdX = np.hstack(np.asarray(
            data["Track"]["Track.mDedx"].array())[eventCuts])[trackCuts]
        pG = np.multiply(np.hstack(pGa), np.hstack(charge))[trackCuts]
        nSigmaProton = np.asarray(data["Track"]["Track.mNSigmaProton"].array())
        nSigmaProtonHisto = np.hstack(np.asarray(
            data["Track"]["Track.mNSigmaProton"].array())[eventCuts])

        ########################################################################
        # Now to get the specific tracks for our net protons. This section of
        # code should be cut out if this is an initial run. Man, I really need
        # to offload this noise to fn.
        pTpro = np.hstack(pT)
        lowPT = np.asarray(np.where((0.4 <= pTpro) & (pTpro < 0.8)))
        highPT = np.asarray(np.where((0.8 <= pTpro) & (pTpro < 2.0)))
        lowPG = np.asarray(np.where(np.hstack(pGa) <= 1.0))
        highPG = np.asarray(np.where(np.hstack(pGa) <= 3.0))
        protonCount = np.asarray(
            np.where(abs(nSigmaProtonHisto - 150.79418624) <= 2))
        uLow, cLow = np.unique(np.hstack((lowPT, lowPG, protonCount)),
                               return_counts=True)
        uHigh, cHigh = np.unique(np.hstack((highPT, highPG, protonCount)),
                                 return_counts=True)
        lowPro = uLow[cLow > 1]
        highPro = uHigh[cHigh > 1]
        break
        for k in range(len(pT)):
            try:
                lowPT = np.asarray(np.where((0.4 <= pT[k]) & (pT[k] < 0.8)))
                highPT = np.asarray(np.where((0.8 <= pT[k]) & (pT[k] < 2.0)))
                lowPG = np.asarray(np.where(pGa[k] <= 1.0))
                highPG = np.asarray(np.where(pGa[k] <= 3.0))
                protonCount = np.asarray(
                    np.where(abs(nSigmaProton[k] - 150.79418624) <= 2))
                uLow, cLow = np.unique(np.hstack((lowPT, lowPG, protonCount)),
                                       return_counts=True)
                uHigh, cHigh = np.unique(np.hstack((highPT, highPG, protonCount)),
                                         return_counts=True)
                lowPro = uLow[cLow > 1]
                highPro = uHigh[cHigh > 1]
                protonsLow = len(np.hstack(np.asarray(
                    np.where(charge[k][lowPro] >= 0))))
                antiprotonsLow = len(
                    np.hstack(np.asarray(np.where(charge[k][lowPro] < 0))))
                protonsHigh = len(np.hstack(np.asarray(
                    np.where(charge[k][highPro] >= 0))))
                antiprotonsHigh = len(
                    np.hstack(np.asarray(np.where(charge[k][highPro] < 0))))
                netProton = protonsLow - antiprotonsLow + protonsHigh - antiprotonsHigh
                if RefMult3[k] > RefCuts[int(len(RefCuts))-1]:
                    netProtonCount[int(len(RefCuts))].append(netProton)
                if RefMult3[k] <= RefCuts[0]:
                    netProtonCount[0].append(netProton)
                for l in range(int(len(RefCuts))-1):
                    if (RefMult3[k] > RefCuts[l]) & (RefMult3[k] <= RefCuts[l+1]):
                        netProtonCount[l+1].append(netProton)
                # This is to map dE/dX to pG and see if we isolated protons.
                counter, bins1, bins2 = np.histogram2d(
                    dEdX[lowPro], pG[lowPro], bins=1000, range=((0, 20), (-5, 5)))
                dEdXpQPRIME1 += counter
                dEdXpQPRIME2 = bins1[:-1]
                dEdXpQPRIME3 = bins2[:-1]
                counter, bins1, bins2 = np.histogram2d(
                    dEdX[highPro], pG[highPro], bins=1000, range=((0, 20), (-5, 5)))
                dEdXpQPRIME1 += counter
                dEdXpQPRIME2 = bins1[:-1]
                dEdXpQPRIME3 = bins2[:-1]
            except IndexError:
                print("Index error in track loop at:", k)
                continue
        ########################################################################

        charge = np.hstack(charge)[trackCuts]
        pT = np.hstack(pT)[trackCuts]

        # Now to get the TOF matched momenta to betas and EPD ring sums.
        pGsmall = []
        for i in range(int(len(eventCuts))):
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
            try:
                pGsmall.append(pGb[i][TofPid[i]])
            except IndexError:  # For if there are no TOF matches in an event.
                print("Index error in TOF/EPD loop at", i)
                break

        beta = np.hstack(beta)
        pGb = np.hstack(np.asarray(pGsmall))
        RingRefPico = np.asarray(RingRefPico)

        # Now to fill our histograms.
        dEdXpQhist = fn.histo2D(dEdX, pG, 20, 5, 0, -5, a, a)
        dEdXpQ = dEdXpQ + dEdXpQhist
        vRhist = fn.histo2D(vY, vX, 10, 10, -10, -10, a, a)
        vR = vR + vRhist
        betaPhist = fn.histo2D(beta, pGb, 10, 10, 0, 0, a, a)
        betaP = betaP + betaPhist
        vZhist = fn.histo1D(vZa, 200, -200, a)
        vZ = vZ + vZhist
        RefMulthist = fn.histo1D(RefMult3, 1000, 0, a)
        RefMult = RefMult + RefMulthist
        for q in range(32):
            RingRefhist = fn.histo1D(RingRefPico[q], 20, 0, b)
            RingRef[q] = RingRef[q] + RingRefhist
        '''
        nSigmaProtonHist = fn.histo1D(nSigmaProtonHisto, 10000, -10000, a)
        nSigmaProtonPico = nSigmaProtonPico + nSigmaProtonHist
        
        counter, bins1, bins2 = np.histogram2d(
            dEdX, pG, bins=1000, range=((0, 20), (-5, 5)))
        dEdXpQPRIME[0] += counter[0]
        dEdXpQPRIME[1] += counter[1]
        dEdXpQPRIME[2] = bins1[:-1]
        dEdXpQPRIME[3] = bins2[:-1]
        '''
        counter, bins = np.histogram(
            nSigmaProtonHisto, bins=1000, range=(-10000, 10000))
        nSigmaProtonPico[0] += counter
        nSigmaProtonPico[1] = bins[:-1]
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
        count += 1
    except IndexError:
        count += 1
        # Identifies the misbehaving file.
        print("IndexError in main loop at", file)
        continue

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
np.save("nSigmaProton.npy", nSigmaProtonPico)
np.save("dEdXpG1.npy", dEdXpQPRIME1)
np.save("dEdXpG2.npy", dEdXpQPRIME2)
np.save("dEdXpG3.npy", dEdXpQPRIME3)
'''
columns = ["vZ", "vZerr", "pT", "pTerr", "vR", "vRerr",
           "Dca", "Dcaerr", "RefMult3", "RefMult3err"]
arr = np.asarray((AveVz[0], AveVz[1], AvePt[0], AvePt[1], AveVr[0], AveVr[1],
                  AveDca[0], AveDca[1], AveRefmult3[0], AveRefmult3[1]))
df = pd.DataFrame(arr.T, index=Runs, columns=columns)
df.to_pickle("averages.pkl")
np.save("vZ.npy", vZ)
np.save("betaP.npy", betaP)
np.save("dEdXpQ.npy", dEdXpQ)
np.save("vR.npy", vR)
np.save("RefMult.npy", RefMult)
np.save("RingRef.npy", RingRef)

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
