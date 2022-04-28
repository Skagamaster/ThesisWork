# Functions for urqmd_reader.py.

import os
import uproot as up
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, moment


class UrQMD:
    """
    pid:
    0 = pi+, 1 = k+, 2 = p, 3 = pi-, 4 = k-, 5 = pbar, 6 = pi0, 7 = eta, 8 = k0, 9 = n, 10 = nbar, 11 = default
    Bill's pid:
    8 = pi+, 9 = pi-, 11 = k+, 12 = k-, 14 = proton, 15 = pbar
    """

    def __init__(self):
        self.b, self.mult, self.pid, self.pt = None, None, None, None
        self.eta, self.phi, self.refmult, self.refmult3 = None, None, None, None
        self.epd_rings, self.protons, self.aprotons, self.nprotons = None, None, None, None

    def get_data(self, file=r"D:\newfile\19\out099.root"):
        with up.open(file) as data:
            pico = data['urqmd']
            self.b = pico['b'].array(library='np')
            self.mult = pico['mul'].array(library='np')
            self.pid = pico['pid'].array()
            self.pt = pico['ptbin'].array() / 100
            self.eta = pico['etabin'].array() * 0.02
            self.phi = pico['phibin'].array() * (np.pi/256.0)
            # refmult includes protons and antiprotons
            refmult = ak.where(((self.pid == 0) | (self.pid == 1) |
                                (self.pid == 3) | (self.pid == 4) |
                                (self.pid == 2) | (self.pid == 5)) &
                               (abs(self.eta) <= 0.5), 1, 0)
            # refmult3 is just charged pions and kaons
            refmult3 = ak.where(((self.pid == 0) | (self.pid == 1) |
                                 (self.pid == 3) | (self.pid == 4)) &
                                (abs(self.eta) <= 1.0), 1, 0)
            protons = ak.where((self.pid == 2) & (abs(self.eta) <= 0.5), 1, 0)
            aprotons = ak.where((self.pid == 5) & (abs(self.eta) <= 0.5), 1, 0)
            index = (self.pt <= 2.0) & (self.pt >= 0.4)
            self.refmult = ak.to_numpy(ak.sum(refmult[index], axis=-1))
            self.refmult3 = ak.to_numpy(ak.sum(refmult3[index], axis=-1))
            self.protons = ak.to_numpy(ak.sum(protons[index], axis=-1))
            self.aprotons = ak.to_numpy(ak.sum(aprotons[index], axis=-1))
            self.nprotons = np.subtract(self.protons, self.aprotons)
            epd_range = [2.14, 2.2, 2.27, 2.34, 2.41, 2.50, 2.59, 2.69, 2.81, 2.94, 3.08, 3.26, 3.47, 3.74,
                         4.03, 4.42, 5.09]
            rings = []
            for i in range(len(epd_range) - 1):
                rings.append(ak.to_numpy(ak.sum(ak.where(((self.pid == 0) | (self.pid == 1) |
                                                          (self.pid == 3) | (self.pid == 4) |
                                                          (self.pid == 2) | (self.pid == 5)) &
                                                         (abs(self.eta) >= epd_range[i]) &
                                                         (abs(self.eta) < epd_range[i + 1]),
                                                         1, 0)[index], axis=-1)))
            self.epd_rings = np.array(rings)


def x_normalise(arr):
    max_val = np.max(arr)
    arr = arr/max_val
    return arr


def cumulants(arr, nprotons, reverse=False, decimals=1):
    arr = np.around(arr, decimals=decimals)
    if reverse is False:
        uniq = np.unique(arr)
    else:
        uniq = np.unique(arr)[::-1]
    cumus = np.empty((4, len(uniq)))
    for i in range(len(uniq)):
        index = arr == uniq[i]
        cumus[0][i] = np.mean(nprotons[index])
        cumus[1][i] = moment(nprotons[index], moment=2)
        cumus[2][i] = moment(nprotons[index], moment=3)
        cumus[3][i] = moment(nprotons[index], moment=4) - 3*np.power(moment(nprotons[index], moment=2), 2)

    return uniq, cumus


def cbwc(refmult, nprotons, centralities):
    """
    WORK IN PROGRESS!!!
    """
    cumu_cbwc = np.zeros((4, 2, len(centralities)+1))
    index = refmult < centralities[0]
    prot = nprotons[index]
    n = len(prot)
    cumu_cbwc[0][0][0] = np.mean(prot)
    cumu_cbwc[0][1][0] = np.std(prot)/np.sqrt(n)
    cumu_cbwc[1][0][0] = moment(prot, moment=2)
    cumu_cbwc[1][1][0] = np.sqrt((moment(prot, moment=4) - moment(prot, moment=2)**2)/n)
    cumu_cbwc[2][0][0] = moment(prot, moment=3)
    cumu_cbwc[2][1][0] = np.sqrt((moment(prot, moment=6) - moment(prot, moment=3)**2 +
                                  9*moment(prot, moment=2)**3 - 6*moment(prot, moment=2)*moment(prot, moment=4))/n)
    cumu_cbwc[3][0][0] = moment(prot, moment=4) - 3*np.power(moment(prot, moment=2), 2)
    cumu_cbwc[3][1][0] = np.sqrt((moment(prot, moment=8) - 12*moment(prot, moment=8)*moment(prot, moment=2) -
                                  8*moment(prot, moment=5)*moment(prot, moment=3) - moment(prot, moment=4)**2 +
                                  48*moment(prot, moment=4)*moment(prot, moment=2)**2 +
                                  64*(moment(prot, moment=3)**2)*moment(prot, moment=2) -
                                  36*moment(prot, moment=2)**2)/n)
    for i in range(len(centralities)-1):
        index = (refmult >= centralities[i]) & (refmult < centralities[i+1])
        prot = nprotons[index]
        n = len(prot)
        cumu_cbwc[0][0][i+1] = np.mean(prot)
        cumu_cbwc[0][1][i+1] = np.std(prot) / np.sqrt(n)
        cumu_cbwc[1][0][i+1] = moment(prot, moment=2)
        cumu_cbwc[1][1][i+1] = np.sqrt((moment(prot, moment=4) - moment(prot, moment=2) ** 2) / n)
        cumu_cbwc[2][0][i+1] = moment(prot, moment=3)
        cumu_cbwc[2][1][i+1] = np.sqrt((moment(prot, moment=6) - moment(prot, moment=3) ** 2 +
                                        9 * moment(prot, moment=2) ** 3 - (6 * moment(prot, moment=2) *
                                        moment(prot, moment=4))) / n)
        cumu_cbwc[3][0][i+1] = moment(prot, moment=4) - 3 * np.power(moment(prot, moment=2), 2)
        cumu_cbwc[3][1][i+1] = np.sqrt((moment(prot, moment=8) - 12 * moment(prot, moment=6) *
                                        moment(prot, moment=2) -
                                        8 * moment(prot, moment=5) * moment(prot, moment=3) -
                                        moment(prot, moment=4) ** 2 +
                                        48 * moment(prot, moment=4) * (moment(prot, moment=2) ** 2) +
                                        64 * (moment(prot, moment=3) ** 2) * moment(prot, moment=2) -
                                        36 * (moment(prot, moment=2) ** 2)) / n)
    index = refmult >= centralities[-1]
    prot = nprotons[index]
    n = len(prot)
    cumu_cbwc[0][0][-1] = np.mean(prot)
    cumu_cbwc[0][1][-1] = np.std(prot)/np.sqrt(n)
    cumu_cbwc[1][0][-1] = moment(prot, moment=2)
    cumu_cbwc[1][1][-1] = np.sqrt((moment(prot, moment=4) - moment(prot, moment=2)**2)/n)
    cumu_cbwc[2][0][-1] = moment(prot, moment=3)
    cumu_cbwc[2][1][-1] = np.sqrt((moment(prot, moment=6) - moment(prot, moment=3)**2 +
                                  9*moment(prot, moment=2)**3 - 6*moment(prot, moment=2)*moment(prot, moment=4))/n)
    cumu_cbwc[3][0][-1] = moment(prot, moment=4) - 3*np.power(moment(prot, moment=2), 2)
    cumu_cbwc[3][1][-1] = np.sqrt((moment(prot, moment=8) - 12*moment(prot, moment=6)*moment(prot, moment=2) -
                                  8*moment(prot, moment=5)*moment(prot, moment=3) - moment(prot, moment=4)**2 +
                                  48*moment(prot, moment=4)*(moment(prot, moment=2)**2) +
                                  64*(moment(prot, moment=3)**2)*moment(prot, moment=2) -
                                  36*(moment(prot, moment=2)**2))/n)
    return cumu_cbwc


def cbwc_b(refmult, nprotons, centralities):
    """
    WORK IN PROGRESS!!!
    """
    cumu_cbwc = np.zeros((4, 2, len(centralities)+1))
    index = refmult >= centralities[0]
    prot = nprotons[index]
    n = len(prot)
    cumu_cbwc[0][0][0] = np.mean(prot)
    cumu_cbwc[0][1][0] = np.std(prot)/np.sqrt(n)
    cumu_cbwc[1][0][0] = moment(prot, moment=2)
    cumu_cbwc[1][1][0] = np.sqrt((moment(prot, moment=4) - moment(prot, moment=2)**2)/n)
    cumu_cbwc[2][0][0] = moment(prot, moment=3)
    cumu_cbwc[2][1][0] = np.sqrt((moment(prot, moment=6) - moment(prot, moment=3)**2 +
                                  9*moment(prot, moment=2)**3 - 6*moment(prot, moment=2)*moment(prot, moment=4))/n)
    cumu_cbwc[3][0][0] = moment(prot, moment=4) - 3*np.power(moment(prot, moment=2), 2)
    cumu_cbwc[3][1][0] = np.sqrt((moment(prot, moment=8) - 12*moment(prot, moment=8)*moment(prot, moment=2) -
                                  8*moment(prot, moment=5)*moment(prot, moment=3) - moment(prot, moment=4)**2 +
                                  48*moment(prot, moment=4)*moment(prot, moment=2)**2 +
                                  64*(moment(prot, moment=3)**2)*moment(prot, moment=2) -
                                  36*moment(prot, moment=2)**2)/n)
    for i in range(len(centralities)-1):
        index = (refmult < centralities[i]) & (refmult >= centralities[i+1])
        prot = nprotons[index]
        n = len(prot)
        cumu_cbwc[0][0][i+1] = np.mean(prot)
        cumu_cbwc[0][1][i+1] = np.std(prot) / np.sqrt(n)
        cumu_cbwc[1][0][i+1] = moment(prot, moment=2)
        cumu_cbwc[1][1][i+1] = np.sqrt((moment(prot, moment=4) - moment(prot, moment=2) ** 2) / n)
        cumu_cbwc[2][0][i+1] = moment(prot, moment=3)
        cumu_cbwc[2][1][i+1] = np.sqrt((moment(prot, moment=6) - moment(prot, moment=3) ** 2 +
                                        9 * moment(prot, moment=2) ** 3 - (6 * moment(prot, moment=2) *
                                        moment(prot, moment=4))) / n)
        cumu_cbwc[3][0][i+1] = moment(prot, moment=4) - 3 * np.power(moment(prot, moment=2), 2)
        cumu_cbwc[3][1][i+1] = np.sqrt((moment(prot, moment=8) - 12 * moment(prot, moment=6) *
                                        moment(prot, moment=2) -
                                        8 * moment(prot, moment=5) * moment(prot, moment=3) -
                                        moment(prot, moment=4) ** 2 +
                                        48 * moment(prot, moment=4) * (moment(prot, moment=2) ** 2) +
                                        64 * (moment(prot, moment=3) ** 2) * moment(prot, moment=2) -
                                        36 * (moment(prot, moment=2) ** 2)) / n)
    index = refmult < centralities[-1]
    prot = nprotons[index]
    n = len(prot)
    cumu_cbwc[0][0][-1] = np.mean(prot)
    cumu_cbwc[0][1][-1] = np.std(prot)/np.sqrt(n)
    cumu_cbwc[1][0][-1] = moment(prot, moment=2)
    cumu_cbwc[1][1][-1] = np.sqrt((moment(prot, moment=4) - moment(prot, moment=2)**2)/n)
    cumu_cbwc[2][0][-1] = moment(prot, moment=3)
    cumu_cbwc[2][1][-1] = np.sqrt((moment(prot, moment=6) - moment(prot, moment=3)**2 +
                                  9*moment(prot, moment=2)**3 - 6*moment(prot, moment=2)*moment(prot, moment=4))/n)
    cumu_cbwc[3][0][-1] = moment(prot, moment=4) - 3*np.power(moment(prot, moment=2), 2)
    cumu_cbwc[3][1][-1] = np.sqrt((moment(prot, moment=8) - 12*moment(prot, moment=6)*moment(prot, moment=2) -
                                  8*moment(prot, moment=5)*moment(prot, moment=3) - moment(prot, moment=4)**2 +
                                  48*moment(prot, moment=4)*(moment(prot, moment=2)**2) +
                                  64*(moment(prot, moment=3)**2)*moment(prot, moment=2) -
                                  36*(moment(prot, moment=2)**2))/n)
    return cumu_cbwc


def b_var(refmult, b, centralities):
    """
    WORK IN PROGRESS!!!
    """
    phi = np.zeros((2, len(centralities)+1))
    index = refmult <= centralities[0]
    prot = b[index]
    n = len(prot)
    phi[0][0] = moment(prot, moment=2)
    phi[1][0] = np.sqrt((moment(prot, moment=4) - moment(prot, moment=2) ** 2) / n)
    for i in range(len(centralities)-1):
        index = (refmult > centralities[i]) & (refmult <= centralities[i+1])
        prot = b[index]
        n = len(prot)
        phi[0][i+1] = moment(prot, moment=2)
        phi[1][i+1] = np.sqrt((moment(prot, moment=4) - moment(prot, moment=2) ** 2) / n)
    index = refmult > centralities[-1]
    prot = b[index]
    n = len(prot)
    phi[0][-1] = moment(prot, moment=2)
    phi[1][-1] = np.sqrt((moment(prot, moment=4) - moment(prot, moment=2) ** 2) / n)
    return phi


def b_var_b(refmult, b, centralities):
    """
    WORK IN PROGRESS!!!
    """
    phi = np.zeros((2, len(centralities)+1))
    index = refmult >= centralities[0]
    prot = b[index]
    n = len(prot)
    phi[0][0] = moment(prot, moment=2)
    phi[1][0] = np.sqrt((moment(prot, moment=4) - moment(prot, moment=2) ** 2) / n)
    for i in range(len(centralities)-1):
        index = (refmult < centralities[i]) & (refmult >= centralities[i+1])
        prot = b[index]
        n = len(prot)
        phi[0][i+1] = moment(prot, moment=2)
        phi[1][i+1] = np.sqrt((moment(prot, moment=4) - moment(prot, moment=2) ** 2) / n)
    index = refmult < centralities[-1]
    prot = b[index]
    n = len(prot)
    phi[0][-1] = moment(prot, moment=2)
    phi[1][-1] = np.sqrt((moment(prot, moment=4) - moment(prot, moment=2) ** 2) / n)
    return phi


def cbwc_dnp(refmult, nprotons, centralities):
    uniq = np.unique(refmult)
    cumus = np.empty((4, len(uniq)))
    cumus_cbwc = np.empty((4, len(centralities)+1))
    n = np.empty(len(uniq))
    for i in range(len(uniq)):
        index = refmult == uniq[i]
        n[i] = len(nprotons[index])
        cumus[0][i] = np.mean(nprotons[index]) * n[i]
        cumus[1][i] = moment(nprotons[index], moment=2) * n[i]
        cumus[2][i] = moment(nprotons[index], moment=3) * n[i]
        cumus[3][i] = moment(nprotons[index], moment=4) - 3 * np.power(moment(nprotons[index], moment=2), 2) * n[i]
    index = uniq < centralities[0]
    for i in range(4):
        cumus_cbwc[i][0] = np.sum(cumus[i][index])/np.sum(n[index])
    for i in range(len(centralities)-1):
        index = (uniq < centralities[i+1]) & (uniq >= centralities[i])
        for j in range(4):
            cumus_cbwc[j][i+1] = np.sum(cumus[j][index]) / np.sum(n[index])
    index = uniq >= centralities[-1]
    for i in range(4):
        cumus_cbwc[i][-1] = np.sum(cumus[i][index]) / np.sum(n[index])
    return cumus_cbwc


def cbwc_dnp_b(refmult, nprotons, centralities, decimals=1):
    refmult = np.around(refmult, decimals=decimals)
    uniq = np.unique(refmult)
    cumus = np.empty((4, len(uniq)))
    cumus_cbwc = np.empty((4, len(centralities)+1))
    n = np.empty(len(uniq))
    for i in range(len(uniq)):
        index = refmult == uniq[i]
        n[i] = len(nprotons[index])
        cumus[0][i] = np.mean(nprotons[index]) * n[i]
        cumus[1][i] = moment(nprotons[index], moment=2) * n[i]
        cumus[2][i] = moment(nprotons[index], moment=3) * n[i]
        cumus[3][i] = moment(nprotons[index], moment=4) - 3 * np.power(moment(nprotons[index], moment=2), 2) * n[i]
    index = uniq >= centralities[0]
    for i in range(4):
        cumus_cbwc[i][0] = np.sum(cumus[i][index])/np.sum(n[index])
    for i in range(len(centralities)-1):
        index = (uniq >= centralities[i+1]) & (uniq < centralities[i])
        for j in range(4):
            cumus_cbwc[j][i+1] = np.sum(cumus[j][index]) / np.sum(n[index])
    index = uniq < centralities[-1]
    for i in range(4):
        cumus_cbwc[i][-1] = np.sum(cumus[i][index]) / np.sum(n[index])
    return cumus_cbwc


def cbwc_b_dnp(refmult, nprotons, centralities):
    pass


def err_add(*args):
    quad = 0
    for a in args:
        quad += a**2
    return np.sqrt(quad)


def err_ratio(q, arr, err_arr):
    q = abs(q)
    rat = q*np.sqrt(np.sum(np.power(np.divide(err_arr, arr), 2)))
    return rat
