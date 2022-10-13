import uproot as up
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
import seaborn as sns
from scipy.signal import argrelextrema as arex
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter as sgf
import awkward as ak


# Savitzky-Golay filter to find the nSigmaProton peak.

pt_vals = [r'$p_T$ = 0.1-0.2 $\frac{GeV}{c}$',
           r'$p_T$ = 0.2-0.3 $\frac{GeV}{c}$',
           r'$p_T$ = 0.3-0.4 $\frac{GeV}{c}$',
           r'$p_T$ = 0.4-0.5 $\frac{GeV}{c}$',
           r'$p_T$ = 0.5-0.6 $\frac{GeV}{c}$',
           r'$p_T$ = 0.6-0.7 $\frac{GeV}{c}$',
           r'$p_T$ = 0.7-0.8 $\frac{GeV}{c}$',
           r'$p_T$ = 0.8-0.9 $\frac{GeV}{c}$',
           r'$p_T$ = 0.9-1.0 $\frac{GeV}{c}$',
           r'$p_T$ = 1.0-1.1 $\frac{GeV}{c}$',
           r'$p_T$ = 1.1-1.2 $\frac{GeV}{c}$']


def savGol(nSigmaProton, xaxis, poly=6, run=123456):
    fig, ax = plt.subplots(3, 4, figsize=(16, 9), constrained_layout=True)
    for i in range(len(nSigmaProton)):
        a = int(i/4)
        b = i % 4
        sg = sgf(nSigmaProton[i], 201, poly)
        maxes = np.asarray(
            xaxis[i][np.hstack(np.asarray(arex(sg, np.greater, order=10)))])
        # Now take the max closest to 0.
        index = np.absolute(maxes) == np.min(np.absolute(maxes))
        ax[a, b].plot(xaxis[i], nSigmaProton[i], color='black', label='raw')
        ax[a, b].plot(xaxis[i], sg, lw=3, color='orange', label="Smoothed")
        for j in maxes:
            ax[a, b].axvline(j, color='red', label=r'$\frac{d^2f}{dx^2}$ max')
        ax[a, b].axvline(maxes[index], color='blue', lw=3, label='p peak')
        ax[a, b].set_xlabel(r'$n\sigma_{p}$', fontsize=12, loc='right')
        ax[a, b].set_ylabel('N', fontsize=12, loc='top')
        ax[a, b].set_title(pt_vals[i], fontsize=15)
    ax[-1, -1].set_axis_off()
    ax[-1, -1].plot(1, c='r', lw=4, label=r'$\frac{d^2f}{dx^2}$ max')
    ax[-1, -1].plot(1, c='blue', lw=4, label=r'p peak')
    ax[-1, -1].plot(1, c='black', lw=4, label=r'raw')
    ax[-1, -1].plot(1, c='orange', lw=4, label=r'smoothed')
    ax[-1, -1].legend(fontsize=20, loc='center')
    fig.suptitle(run, fontsize=20)
    return fig


# This changes the index of any array given.


def indexer(index, *args):
    for arg in args:
        arg = arg[index]
        yield arg


# In case we need Gaussian functions.


def gaus(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def gausDouble(x, x0, x1, a, sigma, b, sigma1):
    return gaus(x, a, x0, sigma) + gaus(x, b, x1, sigma1)

# This will give a list of the charges (as + or - 1) for all tracks.


def chargeList(a):
    # For the charge of a particle (+/- 1).
    a = abs(a)/a
    return a

# This gives us our rapidity.


def rapidity(pZ):
    ePp = np.power(np.add(0.9382720813**2, np.power(pZ, 2)), 1/2)
    eMp = np.power(np.subtract(0.9382720813**2, np.power(pZ, 2)), 1/2)
    y = np.multiply(np.log(np.divide(ePp, eMp)), 1/2)
    ynan = np.isnan(y)
    y[ynan] = 999  # Gets rid of x/0 quantities
    return y


def rapidity_awk(pZ):
    ePp = np.power(np.add(0.9382720813**2, np.power(pZ, 2)), 1/2)
    eMp = np.subtract(0.9382720813**2, np.power(pZ, 2))
    eMp = ak.where(eMp < 0.0, 0.0, eMp)  # to avoid imaginary numbers
    eMp = np.power(eMp, 1/2)
    eMp = ak.where(eMp == 0.0, 1e-10, eMp)  # to avoid infinities
    y = np.multiply(np.log(np.divide(ePp, eMp)), 1/2)
    return y

# ---------------------------------------------------------------------------- #
# EPD ID FUNCTIONS #

# Gets ew, pp, and tt from EPD ID
# ONLY DOES ONE 1D ARRAY RIGHT NOW


def EPDTile(a):
    ew = abs(a)/a
    ew[ew < 0] = 0
    pp = (abs(a/100)).astype(int)
    tt = abs(a) % 100
    return (ew*371 + (pp-1)*31 + tt-1).astype(int)


def EPDTile_awk(a):
    ew = abs(a)/a
    ew = ak.where(ew < 0, 0, ew)
    pp = np.floor(abs(a/100))
    tt = abs(a) % 100
    tiles = np.floor(ew*371 + (pp-1)*31 + tt-1)
    return ak.values_astype(tiles, "int64")


def EPDewpptt_awk(a):
    ew = abs(a)/a
    ew = ak.where(ew < 0, 0, ew)
    pp = np.floor(abs(a/100))
    tt = abs(a) % 100
    return np.vstack((ew, pp, tt))

# Converts epdID to ring position


def epd_adc_awk_old(a):
    counts = ak.num(a)
    a_flat = ak.to_numpy(ak.flatten(a))
    a_dtype = int(str(a_flat.dtype)[-2:])
    raw_bytes = a_flat.view(np.uint8)
    raw_bits = np.unpackbits(raw_bytes, bitorder="big")
    reshaped_bits = raw_bits.reshape(-1, a_dtype)
    truncated_bits = reshaped_bits[:, -13:-1]
    padded_bits = np.pad(truncated_bits, ((0, 0), (0, 4)))
    as_bytes_again = np.packbits(padded_bits, bitorder="big")
    b = ak.unflatten(as_bytes_again.view(">i2"), counts)
    return b

# Returns the ADC values for epdID.


def epd_adc_awk(a):
    counts = ak.num(a)
    a_flat = ak.to_numpy(ak.flatten(a))
    a_flat_ = a_flat.view(np.uint8)[::4]
    b = ak.unflatten(a_flat_, counts)
    return b


def epdRing(a):
    val = np.absolute(a)/a
    row = []
    for x in range(int(len(val))):
        val[x][val[x] < 0] = 0
        row.append(((np.absolute(a[x]) % 100) /
                    2).astype(int)+val[x].astype(int)*16)
    row = np.asarray(row)
    return row

def epdRing_awk(a):
    val = np.absolute(a)/a
    val = ak.where(val < 0, 0, val)
    row = np.floor((np.absolute(a) % 100) / 2) + (val * 16)
    return ak.values_astype(row, "int64")
# ---------------------------------------------------------------------------- #
# FUNCTIONS FOR QA CUTS #

# This creates event cuts based on acceptable vertex parameters.


def vertCuts(vR, vZ, vRval=2.0, vZval=70.0):
    # This is for cuts to be used for all events.
    vRcut = np.hstack(np.asarray(np.where(vR <= vRval)))
    vZcut = np.hstack(np.asarray(
        np.where(abs(vZ) <= vZval)))
    # This is the index list for vR and vZ cuts as above.
    u, c = np.unique(np.hstack((vRcut, vZcut)), return_counts=True)
    return u[c > 1]

# This creates track cuts based on all of the inputs listed.


def trackCuts(Dca, nHitsDedx, nHitsFit, nHitsMax, beta, pG, dEdX,
              DcaVal=1.0, nHitsDedxVal=5, nHitsFitVal=20):
    DcaCut = np.asarray(np.where(Dca <= DcaVal))
    nHitsDedxCut = np.asarray(np.where(nHitsDedx >= nHitsDedxVal))
    nHitsFitCut = np.asarray(np.where(nHitsFit >= nHitsFitVal))
    nHitsMaxCut = np.asarray(np.where(np.divide(nHitsFit, nHitsMax) > 0.52))
    betaCut = np.nonzero(beta)
    # Cutting on dE/dX vs p_G.
    # dEdXpGcut = np.asarray(np.where((dEdX >= np.divide(1, np.power(pG, 2))+2) & (
    #    dEdX <= np.divide(1, np.power(pG - (0.4*np.divide(pG, np.abs(pG))), 2))+3.2)))
    dEdXpGcut = np.asarray(np.where(dEdX <= 3))
    # "Good" track index list.
    u1, c1 = np.unique(np.hstack((DcaCut, nHitsDedxCut, nHitsFitCut,
                                  dEdXpGcut, nHitsMaxCut, betaCut)),
                       return_counts=True)
    return u1[c1 > 1]


# This is to test the trackCuts function.
def trackCutsTest(df, DcaVal=1.0, nHitsDedxVal=5, nHitsFitVal=20):
    DcaCut = np.asarray(np.where(df['Dca'].to_numpy() <= DcaVal))
    nHitsDedxCut = np.asarray(np.where(df['nHitsDedx'].to_numpy() >= nHitsDedxVal))
    nHitsFitCut = np.asarray(np.where(df['nHitsFit'].to_numpy() >= nHitsFitVal))
    nHitsMaxCut = np.asarray(np.where(np.divide(df['nHitsFit'].to_numpy(), df['nHitsMax'].to_numpy()) > 0.52))
    betaCut = np.nonzero(df['beta'].to_numpy())
    # Cutting on dE/dX vs p_G.
    dEdXpGcut = np.asarray(np.where(df['dEdX'].to_numpy() == 0.0))
    # "Good" track index list.
    u1, c1 = np.unique(np.hstack((DcaCut, nHitsDedxCut, nHitsFitCut,
                                  dEdXpGcut, nHitsMaxCut, betaCut)),
                       return_counts=True)
    return u1[c1 > 1]

# This gives us our proton cuts within the "track loop." Defaults are for
# low pT proton cuts.


def protonCuts(pT, pGproton, nSigmaProton, nspMean, pG, pZ,
               pTlow=0.4, pThigh=0.8, pGmax=1.0, rapMax=0.5):
    rapidit = rapidity(pZ)
    rapidCut = np.asarray(np.where(np.absolute(rapidit) <= rapMax))
    PT = np.asarray(np.where((pTlow <= pT) & (pT < pThigh)))
    PG = np.asarray(np.where(np.absolute(pGproton) <= pGmax))
    protonCount = np.asarray(
        np.where(np.absolute(nSigmaProton - nspMean) <= 2000))
    uP, cP = np.unique(np.hstack((PT, PG, protonCount, rapidCut)),
                       return_counts=True)
    Pro = uP[cP > 1]
    print(PT, PG, rapidCut)
    return Pro

# Proton cuts with the pG ranges included for the means.


def protonCutsBeta(pT, pGproton, nSigmaProton, nspMean1, pG, pZ, pRange,
                   pTlow=0.4, pThigh=0.8, pGmax=1.0, rapMax=0.5, i=3):
    rapidit = rapidity(pZ)
    rapidCut = np.asarray(np.where(np.absolute(rapidit) <= rapMax))
    PT = np.asarray(np.where((pTlow <= pT) & (pT < pThigh)))
    PG = np.asarray(np.where(np.absolute(pGproton) <= pGmax))
    boolGuy = False
    if np.max(pRange) > pGmax:
        pRangeCut = np.asarray(np.where(pRange <= pGmax))
        nspMean1 = nspMean1[pRangeCut]
    else:
        boolGuy = True
    protonCount = []
    for y in range(int(len(nspMean1)-1)):
        protonCount.append([])
        PG1 = np.asarray(np.where((np.absolute(pGproton) >= pRange[y]) &
                                  (np.absolute(pGproton) < pRange[y+1])))
        nSig = np.asarray(np.where(np.absolute(
            nSigmaProton - nspMean1[y]) <= 2000))
        try:
            u1, c1 = np.unique(np.hstack((PG1, nSig)), return_counts=True)
            protonCount[-1] = u1[c1 > 1]
        except Exception as e:
            print("Darn it!", e.__class__, "occurred.")
        '''
        if (nSig.size != 0) & (PG1.size != 0):
            u1, c1 = np.unique(np.hstack((PG1, nSig)), return_counts=True)
            protonCount[-1] = u1[c1 > 1]
        '''
    if boolGuy is True:
        protonCount.append([])
        PG1 = np.asarray(np.where(np.absolute(PG) >= np.max(nspMean1)))[1]
        nSig = np.asarray(np.where(np.absolute(nSigmaProton) <= 2000))[0]
        try:
            u1, c1 = np.unique(np.hstack((PG1, nSig)), return_counts=True)
            protonCount[-1] = u1[c1 > 1]
        except Exception as e:
            print("Shoot!", e.__class__, "occurred.")
        '''
        if (nSig.size != 0) & (PG1.size != 0):
            u1, c1 = np.unique(np.hstack((PG1, nSig)), return_counts=True)
            protonCount[-1] = u1[c1 > 1]
        '''
    protonCount = np.asarray(protonCount)
    if protonCount.size == 0:
        protonCount = np.array([[]])
    else:
        protonCount = np.asarray(
            [np.hstack(np.asarray(protonCount))])
    uP, cP = np.unique(np.hstack((PT, PG, protonCount, rapidCut)),
                       return_counts=True)
    Pro = uP[cP > 1]
    Pro = Pro.astype(int)
    return Pro
# ---------------------------------------------------------------------------- #
