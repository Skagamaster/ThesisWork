# Original code:
# \authors Chad Rexrode & Jennifer Klay
# \date 06/13/2014
# \email jklay@calpoly.edu
# \affiliation California Polytechnic State University
#
# Code updates:
# \author Skipper Kagamaster
# \date 07/22/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
#
import IPython.core.pylabtools as pyt
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import urllib.request as urlr
from scipy.optimize import curve_fit
import time
from numpy.random import default_rng
rng = default_rng()

# Importing data from particle data group
# Attempts to use data from current year
# If that data is not available, drops down a year until data is found or defaults to 2013 data
DataFound1 = True
DataFound2 = True
Y = date.today().year
count_1 = 0
while count_1 <= Y - 2013:
    try:
        TotalData = urlr.urlopen(
            'http://pdg.lbl.gov/' + str(Y - count_1) + '/hadronic-xsections/rpp' + str(Y - count_1) + '-pp_total.dat')
        print("Using " + str(Y - count_1) + " data for total cross section.")
        break
    except:
        print(str(Y - count_1) +
              "total cross section data is unavailable. The Particle Data Group website may not have the "
              "latest data or may have changed format.")
        count_1 += 1
        if count_1 > Y - 2013:
            print(
                "---\nData not found. Please check your internet connection to "
                "http://pdg.lbl.gov/2013/html/computer_read.html\n---")
            DataFound1 = False
count_2 = 0
while count_2 <= Y - 2013:
    try:
        ElasticData = urlr.urlopen(
            'http://pdg.lbl.gov/' + str(Y - count_2) + '/hadronic-xsections/rpp' + str(Y - count_2) + '-pp_elastic.dat')
        print("Using " + str(Y - count_2) + " data for elastic cross section.")
        break
    except:
        count_2 += 1
        if count_2 > Y - 2013:
            print(
                "---\nData not found. Please check your internet connection to "
                "http://pdg.lbl.gov/2013/html/computer_read.html\n---")
            DataFound2 = False

if DataFound1:
    data = np.loadtxt(TotalData, float, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8), skiprows=11)
    Point = data[:, 0]
    Plab = data[:, 1]  # GeV/c
    Plab_min = data[:, 2]
    Plab_max = data[:, 3]
    Sig = data[:, 4]
    StEr_H = data[:, 5]
    StEr_L = data[:, 6]
    SyEr_H = data[:, 7]
    SyEr_L = data[:, 8]
if DataFound2:
    Edata = np.loadtxt(ElasticData, float, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8), skiprows=11)
    EPoint = Edata[:, 0]
    EPlab = Edata[:, 1]  # GeV/c
    EPlab_min = Edata[:, 2]
    EPlab_max = Edata[:, 3]
    ESig = Edata[:, 4]
    EStEr_H = Edata[:, 5]
    EStEr_L = Edata[:, 6]
    ESyEr_H = Edata[:, 7]
    ESyEr_L = Edata[:, 8]

pyt.figsize(12, 7)


def Ecm(Plab):
    """Converts Plab momenta to center of mass energy [GeV]."""
    E = (((Plab ** 2 + .938 ** 2) ** (1 / 2.) + .938) ** 2 - (Plab ** 2)) ** (1 / 2.)
    return E


if DataFound1 and DataFound2:
    # Automatically converts all P_lab momenta to corresponding center-of-mass energy [GeV]
    E_cm = Ecm(Plab)
    eE_cm = Ecm(EPlab)
    cm_min = Ecm(Plab_min)
    cm_max = Ecm(Plab_max)
    ecm_min = Ecm(EPlab_min)
    ecm_max = Ecm(EPlab_max)


# Define best fit curve given by the particle data group
def func(S, P, H, M, R1, R2, n1, n2):
    m = .93827  # Proton mass GeV/c^2
    sM = (2 * m + M) ** 2  # Mass^2 (GeV/c^2)^2
    hbar = 6.58211928 * 10 ** -25  # GeV*s
    c = 2.99792458 * 10 ** 8  # m/s
    with np.errstate(all='ignore'):  # TODO Actually fix the problem instead of ignoring it.
        sigma = H * (np.log(S ** 2 / sM)) ** 2 + P + R1 * (S ** 2 / sM) ** (-n1) - R2 * (S ** 2 / sM) ** (-n2)
    return sigma


# Apply best fit curve to the elastic cross-section data
s = eE_cm[:]
y = ESig[:]
p0 = [4.45, .0965, 2.127, 11, 4, .55, .55]
popt, pcov = curve_fit(func, s, y, p0)

# Apply best fit curve to total cross-section data
s2 = E_cm[90:]
y2 = Sig[90:]
p0 = [34.49, .2704, 2.127, 12.98, 7.38, .451, .549]
popt2, pcov2 = curve_fit(func, s2, y2, p0)


def SigI(BE):
    """Returns the proton-proton cross-sectional area [fm^2] for given beam energy [GeV]"""
    return .1 * (func(BE, popt2[0], popt2[1], popt2[2], popt2[3], popt2[4], popt2[5], popt2[6]) -
                 func(BE, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6]))


def DisplayData():
    """Displays the Proton-Proton Cross Section Data."""
    plt.loglog(E_cm, Sig, ls=' ', marker='.', markersize=3, color='black', label='PDG Data')
    plt.loglog(eE_cm, ESig, ls=' ', marker='.', markersize=3, color='black')
    plt.loglog(E_cm[90:], func(E_cm[90:], popt2[0], popt2[1], popt2[2], popt2[3], popt2[4], popt2[5], popt2[6]),
               color='blue')
    plt.loglog(E_cm, func(E_cm, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6]), color='blue',
               label='Fit')
    plt.scatter(7000, 70.5, label='TOTEM EPL 101 21003', color='red')
    plt.scatter([2760, 7000], [62.1, 72.7], label='ALICE 2013 EPJ C73', color='blue')
    plt.scatter([7000, 8000], [72.9, 74.7], label='TOTEM 2013 PRL 111 012001 ', color='green')
    plt.errorbar([2760, 7000, 7000, 7000, 8000], [62.1, 70.5, 72.7, 72.9, 74.7], yerr=[5.9, 3.4, 6.2, 1.5, 1.7],
                 fmt=' ', color='black')
    plt.loglog(E_cm[90:], 10 * SigI(E_cm[90:]))
    #plt.errorbar(E_cm, Sig, xerr=[E_cm - cm_min, cm_max - E_cm], yerr=[StEr_L, StEr_H], ms=.5, mew=0, fmt=None,
    #             ecolor='black')
    #plt.errorbar(eE_cm, ESig, xerr=[eE_cm - ecm_min, ecm_max - eE_cm], yerr=[EStEr_L, EStEr_H], ms=.5, mew=0, fmt=None,
    #             ecolor='black')
    plt.annotate("Total", fontsize=11, xy=(7, 46), xytext=(7, 46))
    plt.annotate("Elastic", fontsize=11, xy=(1000, 10), xytext=(1000, 10))
    plt.annotate("Inelastic", fontsize=11, xy=(35, 25), xytext=(35, 25))
    plt.title("pp Cross Section Data", fontsize=16)
    plt.ylabel("Cross Section [mb]", fontsize=12)
    plt.xlabel("$\sqrt{s}\,[GeV]$", fontsize=16)
    plt.ylim(1, 400)
    plt.grid(which='minor', axis='y')
    plt.grid(which='major', axis='x')
    plt.legend(loc=4)
    plt.show()


# Reads in parameters to calculate nuclear charge densities (NCD)
parameters = np.loadtxt("WoodSaxonParameters.txt", str, delimiter='\t')
FBdata = np.loadtxt("FourierBesselParameters.txt", str, delimiter='\t')
pNucleus = parameters[:, 0]
pModel = parameters[:, 1]
pr2 = parameters[:, 2]
pC_A = parameters[:, 3]
pZ_Alpha = parameters[:, 4]
pw = parameters[:, 5]
FBnucleus = FBdata[:, 0]
FBrms = FBdata[:, 1]
FBR = FBdata[:, 2]
FBa = np.zeros((len(FBnucleus) - 1, 17), float)
for i in range(len(FBnucleus) - 1):
    FBa[i, :] = FBdata[i + 1, 3:]
FBa = FBa.astype(np.float)


def NCD(Nucleus, Model, Range=2, Bins=100):
    """Returns the Nuclear Charge Distribution for a given Nucleus with specified
    model. Creates radial distribution from 0 to Range*nuclear radius with n
    number of bins. If no values are set, defaults to 197Au using two-parameter
    Fermi model up to twice the nuclear radius with 100 bins.
     Params:
         Nucleus: The species being generated
         Model: The model used to simulate the nuclear makeup
         Range: Number of radii to include in the scope
         Bins: How many radial bins to include"""
    # For multiple models of the same nucleus takes the first set of parameters and notifies the user which
    # parameters are used.
    j = []
    for index in range(len(pNucleus)):
        if pNucleus[index] == Nucleus and pModel[index] == Model:
            j.append(index)
            i = index
    j = np.array(j, dtype=int)
    if len(j) > 1:
        # print "Multiple parameters detected for given model. Using primary values."
        i = j[0]
    r = np.linspace(0, Range * float(pr2[i]), Bins)
    if Model == 'HO':
        return (1 + float(pZ_Alpha[i]) * (r / float(pC_A[i])) ** 2) * np.exp(-1 * (r / float(pC_A[i])) ** 2)
    elif Model == 'MHO':
        print("Warning: Model not yet supported\nPlease use a different model.")
        return None
    elif Model == 'Mi':
        print("Warning: Model not yet supported\nPlease use a different model.")
        return None
    elif Model == 'FB':
        # print "Warning: Fourier-Bessel Model currently contains support for H-3, He-3, C-12, and O-16 only.
        # If not these ions, please choose another model."
        '''
        for FBindex in range(len(FBnucleus)):
            if FBnucleus[FBindex] == Nucleus:
                iFB = FBindex
        '''
        iFB = (FBnucleus == Nucleus)
        iFB = np.hstack(np.where(iFB == True))
        r = np.linspace(0, float(FBR[iFB]), Bins)
        p = np.zeros(np.size(r), float)
        v = np.arange(0, 17, 1)
        for k in range(len(r)):
            p[k] = abs(sum(FBa[iFB - 1, v] * np.sinc((v + 1) * r[k] / float(FBR[iFB]))))
        return p
    elif Model == 'SOG':
        print("Warning: Model not yet supported\nPlease use a different model.")
        return None
    elif Model == '2pF':
        return 1 / (1 + np.exp((r - float(pC_A[i])) / float(pZ_Alpha[i])))
    elif Model == '3pF':
        return (1 + float(pw[i]) * r ** 2 / float(pC_A[i]) ** 2) / (
                    1 + np.exp((r - float(pC_A[i])) / float(pZ_Alpha[i])))
    elif Model == '3pG':
        return (1 + float(pw[i]) * r ** 2 / float(pC_A[i]) ** 2) / (
                    1 + np.exp((r ** 2 - float(pC_A[i]) ** 2) / float(pZ_Alpha[i]) ** 2))
    elif Model == 'UG':
        print("Warning: Model not yet supported\nPlease use a different model.")
        return None
    else:
        print('Error: Model not found\nPlease check that the model was typed in correctly. (Case Sensitive)')
        return None


def distribute1D(x, prob, N):
    """Takes any numerical distribution probability, on
    the interval defined by the array x, and returns an
    array of N sampled values that are statistically the
    same as the input data.
     Params:
         x: Range over which to sample data
         prob: Probability distribution
         N: Number of return values to use"""
    y = prob * 4 * np.pi * x ** 2
    A = np.cumsum(y) / (np.cumsum(y)[-1])
    z = rng.random(N)
    B = np.searchsorted(A, z)
    return x[B]


def Collider(N, Particle1, A1, Particle2, A2, model1, model2, Energy, bRange=1.6, Range=2, Bins=100):
    """
    Simulates N collisions between specified Elements (with number of nucleons A)
    and using Center of Mass Energy [GeV].
    Returns the matrices corresponding to center-to-center seperation distance
    (Random value between 0 and bRange*(radius of nucleus1 + radius of nucleus2) [fm]),
    Nuclei 1 and 2, number of participating nucleons, and the number of binary
    collisions. Additionally returns the interaction distance of the nucleons
    given the chosen beam energy and the radii of both nuclei.
     Params:
        N: Number of collisions
        Particle1: Species of the first nucleus
        A1: Atomic number for Particle1
        Particle2: Species for the second nucleus
        A2: Atomic number of Particle2
        model1: Nuclear model for Particle1
        model2: Nuclear model for Particle2
        Energy: COM collision energy
        bRange: How far out, in combined radii, the impact parameter goes
        Range: Radial range in which to place nucleons in the nucleus
        Bins: Number of bins for radial distribution generation"""
    print("Starting the collider. Hold on, Flash ...")

    # Set Rp1 and Rp2 equal to the radii of the nuclei choosen
    # j1 = []
    # j2 = []
    i1 = "Unassigned"
    i2 = "Unassigned"
    j1 = ((pNucleus == Particle1) & (pModel == model1))
    j2 = ((pNucleus == Particle2) & (pModel == model2))
    j1 = np.hstack(np.where(j1 == True))
    j2 = np.hstack(np.where(j2 == True))
    if len(j1) > 0:
        i1 = j1[0]
    if len(j2) > 0:
        i2 = j2[0]

    if len(j1) > 1 or len(j2) > 1:
        print("Multiple parameters detected for specified model. Using primary values.")
        # i1 = j1[0]
        # i2 = j2[0]
    if i1 == "Unassigned" or i2 == "Unassigned":
        print('Error: Model not found\nPlease check that the model was typed in correctly. (Case Sensitive)')
        return None, None, None, None, None, None, None, None
    if model1 != 'FB':
        Rp1 = float(pr2[i1])
    else:
        iFB = (FBnucleus == Particle1)
        iFB = np.hstack(np.where(iFB == True))[-1]
        Rp1 = float(FBR[iFB])
    if model2 != 'FB':
        Rp2 = float(pr2[i2])
    else:
        iFB = (FBnucleus == Particle2)
        iFB = np.hstack(np.where(iFB == True))[-1]
        Rp2 = float(FBR[iFB])
    print(Rp1, Rp2)
    # Create a weighted, random array of impact parameters (where it is more likely to
    # have a peripheral rather than a central collision).
    b_range = np.linspace(0, (Rp1 + Rp2) * bRange, 1000)  # 1000 possible b values to pull from
    b_prob = b_range/np.linalg.norm(b_range, 1)  # linear weighting
    b = rng.choice(b_range, N, p=b_prob)
    # b = (Rp1 + Rp2) * bRange * np.random.random_sample(N)  # Create array of random impact parameters
    if model1 != 'FB':
        r1 = np.linspace(0, Range * Rp1, Bins)  # Array of radial data used for plotting
    else:
        r1 = np.linspace(0, Rp1, Bins)
    if model2 != 'FB':
        r2 = np.linspace(0, Range * Rp2, Bins)
    else:
        r2 = np.linspace(0, Rp2, Bins)
    Npart = np.zeros(N, float)
    Ncoll = np.zeros(N, float)
    Maxr = np.sqrt(SigI(Energy) / np.pi)  # Radius within which two nucleons will interact
    # Runs N number of times; each run fills that iteration of Ncoll, Npart.
    for L in range(N):
        if L % 500 == 499:
            print("Ion collision", L+1, "of", N, "in progress.")
        Nucleus1 = np.zeros((A1, 7), float)
        Nucleus2 = np.zeros((A2, 7), float)
        # Gives each nucleon is own radial distance from the center of the nucleus
        # Sampled from the NCD function and distributed with a factor 4*pi*r^2*p(r)
        Nucleus1[:, 0] = distribute1D(r1, NCD(Particle1, model1, Range, Bins), A1)
        Nucleus2[:, 0] = distribute1D(r2, NCD(Particle2, model2, Range, Bins), A2)
        # Nucleons are then given random azimuthal distances such that no particular point is more likely to be
        # populated than another.
        # Theta coordinates
        Nucleus1[:, 1] = np.arccos(np.multiply(2, rng.random(A1))-1)
        Nucleus2[:, 1] = np.arccos(np.multiply(2, rng.random(A2))-1)
        # Phi coordinates
        Nucleus1[:, 2] = np.multiply(2 * np.pi, rng.random(A1))
        Nucleus2[:, 2] = np.multiply(2 * np.pi, rng.random(A2))
        # Cartesian coordinates (x,y,z) are determined from spherical coordinates and passed to the nuclei arrays.
        # This just makes the distance calculations easier later.
        # x coordinates
        Nucleus1[:, 3] = np.multiply(np.multiply(Nucleus1[:, 0], np.sin(Nucleus1[:, 1])), np.cos(Nucleus1[:, 2]))
        Nucleus2[:, 3] = np.multiply(np.multiply(Nucleus2[:, 0], np.sin(Nucleus2[:, 1])), np.cos(Nucleus2[:, 2]))
        # y coordinates
        Nucleus1[:, 4] = np.multiply(np.multiply(Nucleus1[:, 0], np.sin(Nucleus1[:, 1])), np.sin(Nucleus1[:, 2]))
        Nucleus2[:, 4] = np.multiply(np.multiply(Nucleus2[:, 0], np.sin(Nucleus2[:, 1])), np.sin(Nucleus2[:, 2]))
        # z coordinates
        Nucleus1[:, 5] = np.multiply(Nucleus1[:, 0], np.cos(Nucleus1[:, 1]))
        Nucleus2[:, 5] = np.multiply(Nucleus2[:, 0], np.cos(Nucleus2[:, 1]))

        deltas1 = np.sqrt(np.power((Nucleus1[:, 3] - Nucleus1[:, 3]), 2) +
                          np.power((Nucleus1[:, 4] + Nucleus1[:, 4]), 2) +
                          np.power((Nucleus1[:, 5] + Nucleus1[:, 5]), 2))
        deltas2 = np.sqrt(np.power((Nucleus2[:, 3] - Nucleus2[:, 3]), 2) +
                          np.power((Nucleus2[:, 4] + Nucleus2[:, 4]), 2) +
                          np.power((Nucleus2[:, 5] + Nucleus2[:, 5]), 2))
        failsafe = 10
        failure = 0
        while np.any(deltas1) < Maxr:
            too_close = len(deltas1[deltas1 < Maxr])
            new_rands1 = np.arccos(np.multiply(2, rng.random(A1)) - 1)
            new_rands2 = np.multiply(2 * np.pi, rng.random(A1))
            Nucleus1[:, 1] = np.where(deltas1 < Maxr, new_rands1, Nucleus1[:, 1])
            Nucleus1[:, 2] = np.where(deltas1 < Maxr, new_rands2, Nucleus1[:, 2])
            Nucleus1[:, 3] = np.multiply(np.multiply(Nucleus1[:, 0], np.sin(Nucleus1[:, 1])),
                                         np.cos(Nucleus1[:, 2]))
            Nucleus1[:, 4] = np.multiply(np.multiply(Nucleus1[:, 0], np.sin(Nucleus1[:, 1])),
                                         np.sin(Nucleus1[:, 2]))
            Nucleus1[:, 5] = np.multiply(Nucleus1[:, 0], np.cos(Nucleus1[:, 1]))
            deltas1 = np.sqrt(np.power((Nucleus1[:, 3] - Nucleus1[:, 3]), 2) +
                              np.power((Nucleus1[:, 4] + Nucleus1[:, 4]), 2) +
                              np.power((Nucleus1[:, 5] + Nucleus1[:, 5]), 2))
            failure += 1
            if failure > failsafe:
                Nucleus1[:, 0] = np.where(deltas1 < Maxr, distribute1D(r1, NCD(Particle1, model1, Range, Bins), A1),
                                          Nucleus1[:, 0])
                break
        failure = 0
        while np.any(deltas2) < Maxr:
            too_close = len(deltas2[deltas2 < Maxr])
            new_rands1 = np.arccos(np.multiply(2, rng.random(A2)) - 1)
            new_rands2 = np.multiply(2 * np.pi, rng.random(A2))
            Nucleus2[:, 1] = np.where(deltas2 < Maxr, new_rands1, Nucleus2[:, 1])
            Nucleus2[:, 2] = np.where(deltas2 < Maxr, new_rands2, Nucleus2[:, 2])
            Nucleus2[:, 3] = np.multiply(np.multiply(Nucleus2[:, 0], np.sin(Nucleus2[:, 1])),
                                         np.cos(Nucleus2[:, 2]))
            Nucleus2[:, 4] = np.multiply(np.multiply(Nucleus2[:, 0], np.sin(Nucleus2[:, 1])),
                                         np.sin(Nucleus2[:, 2]))
            Nucleus2[:, 5] = np.multiply(Nucleus2[:, 0], np.cos(Nucleus2[:, 1]))
            deltas2 = np.sqrt(np.power((Nucleus2[:, 3] - Nucleus2[:, 3]), 2) +
                              np.power((Nucleus2[:, 4] + Nucleus2[:, 4]), 2) +
                              np.power((Nucleus2[:, 5] + Nucleus2[:, 5]), 2))
            failure += 1
            if failure > failsafe:
                Nucleus2[:, 0] = np.where(deltas2 < Maxr, distribute1D(r2, NCD(Particle2, model2, Range, Bins), A2),
                                          Nucleus2[:, 0])
                break
        """
        # Now we want to make sure the nucleus does not have overlapping necleons. This can easily be done with a loop,
        # but that's devastatingly slow for any real Monte Carlo numbers (like on the order of 10^6+).
        # TODO Get rid of this loop!
        for p1 in range(A1):
            for p1x in range(A1):
                FailSafe = 0  # Prevents program from running indefinitely in some cases
                if p1x == p1:
                    pass
                else:
                    while np.sqrt(
                            (Nucleus1[p1, 3] - Nucleus1[p1x, 3]) ** 2 + (Nucleus1[p1, 4] + Nucleus1[p1x, 4]) ** 2 + (
                                    Nucleus1[p1, 5] + Nucleus1[p1x, 5]) ** 2) < Maxr:
                        Nucleus1[p1x, 1] = np.arccos(2 * np.random.random_sample(1) - 1)
                        Nucleus1[p1x, 2] = 2 * np.pi * np.random.random_sample(1)
                        Nucleus1[p1x, 3] = Nucleus1[p1x, 0] * np.sin(Nucleus1[p1x, 1]) * np.cos(Nucleus1[p1x, 2])
                        Nucleus1[p1x, 4] = Nucleus1[p1x, 0] * np.sin(Nucleus1[p1x, 1]) * np.sin(Nucleus1[p1x, 2])
                        Nucleus1[p1x, 5] = Nucleus1[p1x, 0] * np.cos(Nucleus1[p1x, 1])
                        FailSafe += 1
                        if FailSafe > 10:
                            Nucleus1[p1x, 0] = distribute1D(r1, NCD(Particle1, model1, Range, Bins), A1)[p1x]
        for p2 in range(A2):
            for p2x in range(A2):
                FailSafe = 0
                if p2x == p2:
                    pass
                else:
                    while np.sqrt(
                            (Nucleus2[p2, 3] - Nucleus2[p2x, 3]) ** 2 + (Nucleus2[p2, 4] - Nucleus2[p2x, 4]) ** 2 + (
                                    Nucleus2[p2, 5] - Nucleus2[p2x, 5]) ** 2) < Maxr:
                        Nucleus2[p2x, 1] = np.arccos(2 * np.random.random_sample(1) - 1)
                        Nucleus2[p2x, 2] = 2 * np.pi * np.random.random_sample(1)
                        Nucleus2[p2x, 3] = Nucleus2[p2x, 0] * np.sin(Nucleus2[p2x, 1]) * np.cos(Nucleus2[p2x, 2])
                        Nucleus2[p2x, 4] = Nucleus2[p2x, 0] * np.sin(Nucleus2[p2x, 1]) * np.sin(Nucleus2[p2x, 2])
                        Nucleus2[p2x, 5] = Nucleus2[p2x, 0] * np.cos(Nucleus2[p2x, 1])
                        FailSafe += 1
                        if FailSafe > 10:
                            Nucleus2[p2x, 0] = distribute1D(r2, NCD(Particle2, model2, Range, Bins), A2)[p2x]
        """
        # Here is where we see how many collisions and participants there were.
        # Create matrices of the x/y coordinates in the transverse plane.
        nuctile1 = np.tile(Nucleus1[:, 3], (A2, 1)).T
        nuctile2 = np.tile(Nucleus2[:, 3], (A1, 1))
        nuctile3 = np.tile(Nucleus1[:, 4], (A2, 1)).T
        nuctile4 = np.tile(Nucleus2[:, 4], (A1, 1))
        # Now we see if there is any overlap.
        Ncoll_builder = np.sqrt(np.add(np.power(np.subtract(nuctile1, nuctile2), 2),
                                       np.power(np.add(-b[L] - nuctile4, nuctile3), 2)))
        Ncoll_builder = np.where(Ncoll_builder.T <= Maxr, 1, 0)
        Ncoll[L] = np.sum(Ncoll_builder)  # Sum of all overlapping nucleons.
        # We are looking for rows/columns in the above matricies which have no collisions.
        # A row/column with no collisions represents a nucleon that did not interact.
        Nucleus1[:, 6] = np.where(np.sum(Ncoll_builder, axis=0) == 0, 1, 0)
        Nucleus2[:, 6] = np.where(np.sum(Ncoll_builder, axis=1) == 0, 1, 0)
        # Number of participating particles is the total number of particles minus all the flagged unwounded particles
        Npart[L] = A1 + A2 - np.sum(Nucleus1[:, 6]) - np.sum(Nucleus2[:, 6])

        if model1 == 'FB':
            Rp1 = float(pr2[i1])
        if model2 == 'FB':
            Rp2 = float(pr2[i2])

    print("And we're all done colliding. Did we time travel?")

    return b, Nucleus1, Nucleus2, Npart, Ncoll, Maxr, Rp1, Rp2


def PlotNuclei(Nucleus1, Nucleus2, Particle1, Particle2, model1, model2, Rp1, Rp2, Range, Bins):
    """
    Plots the nuclear charge density for each nucleus and
    shows the root-mean-square radius.
    Blue corresponds to nucleus 1 and green to nucleus 2.
    """
    r1 = np.linspace(0, Range * Rp1, Bins)
    r2 = np.linspace(0, Range * Rp2, Bins)
    if model1 != 'FB':
        Range1 = Range
    else:
        iFB = np.hstack(np.where(FBnucleus == Particle1))
        RpFB1 = float(FBR[iFB])
        Range1 = 1
        r1 = np.linspace(0, RpFB1, Bins)
    if model2 != 'FB':
        Range2 = Range
    else:
        iFB = np.hstack(np.where(FBnucleus == Particle2))
        RpFB2 = float(FBR[iFB])
        Range2 = 1
        r2 = np.linspace(0, RpFB2, Bins)
    d1 = NCD(Particle1, model1, Range, Bins) / max(NCD(Particle1, model1, Range, Bins))
    d2 = NCD(Particle2, model2, Range, Bins) / max(NCD(Particle2, model2, Range, Bins))
    rms1 = r1[np.abs(r1 - Rp1).argmin()]
    rms2 = r2[np.abs(r2 - Rp2).argmin()]
    plt.plot(r1, d1, color='blue', lw=2.5, label=str(Particle1) + " Radial Density")
    plt.plot(r2, d2, color='green', lw=2.5, label=str(Particle2) + " Radial Density")
    plt.plot([rms1, rms1], [0, d1[np.abs(r1 - Rp1).argmin()]], 'b--', lw=2, label=str(Particle1) + ' rms radius')
    plt.plot([rms2, rms2], [0, d2[np.abs(r2 - Rp2).argmin()]], 'g--', lw=2, label=str(Particle2) + ' rms radius')
    plt.xlabel("Radial Distance [fm]", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend()
    plt.show()


def ShowCollision(N, Particle1, A1, Particle2, A2, Rp1, Rp2, Nucleus1, Nucleus2, b, Npart, Ncoll, Maxr):
    """Plots a cross-sectional and horizontal view of the last collision."""
    fig, ax = plt.subplots()
    N1 = plt.Circle((Rp1, Rp1), Rp1, color='b', fill=False, lw=2)
    N2 = plt.Circle((Rp1 + b[N - 1], Rp1), Rp2, color='g', fill=False, lw=2)
    fig.gca().add_artist(N1)
    fig.gca().add_artist(N2)
    for i in range(A1):
        if Nucleus1[i, 6] == 1:
            ax.plot(Rp1 + Nucleus1[i, 4], Rp1 + Nucleus1[i, 3], 'b.', ms=26, alpha=.6, mew=0, mec='blue')
        if Nucleus1[i, 6] == 0:
            ax.plot(Rp1 + Nucleus1[i, 4], Rp1 + Nucleus1[i, 3], 'r.', ms=26, alpha=.6, mew=0, mec='red')
    for i in range(A2):
        if Nucleus2[i, 6] == 1:
            ax.plot(b[N - 1] + Rp1 + Nucleus2[i, 4], Rp1 + Nucleus2[i, 3], 'g.', ms=26, alpha=.6, mew=0, mec='green')
        if Nucleus2[i, 6] == 0:
            ax.plot(b[N - 1] + Rp1 + Nucleus2[i, 4], Rp1 + Nucleus2[i, 3], 'y.', ms=26, alpha=.6, mew=0, mec='yellow')
    zed = 1.5 * (Rp1 + Rp2) + b[N - 1]
    ax.annotate('Npart=' + str(Npart[N - 1]) + '\nNcoll=' + str(Ncoll[N - 1]), xy=(1, 0), xytext=(0, 1.015 * zed),
                fontsize=16)
    # ax.annotate('Maxr: ' + str(Maxr)[:5] + ' fm', xy=(0, 2 * Rp1), xytext=(.01 * zed, .95 * zed), fontsize=12)
    ax.annotate('b: ' + f'{b[-1]:.2f}' + ' fm', xy=(0, 2 * Rp1), xytext=(.01 * zed, .95 * zed), fontsize=12)
    ax.plot([(.01 * zed), (.01 * zed) + Maxr], [zed * .93, zed * .93], color='r', ls='-', lw=3)
    plt.xlim((-zed/2, zed))
    plt.ylim((-zed/2, zed))
    plt.xlabel('Horizontal Cross Section [fm]', fontsize=15)
    plt.ylabel('Vertical Position [fm]', fontsize=15)
    fig.set_size_inches(6, 6)
    fig1, ax1 = plt.subplots()
    N3 = plt.Circle((Rp1, Rp1), Rp1, color='b', fill=False, lw=2)
    N4 = plt.Circle(((Rp1 + Rp2) * 2, Rp1 + b[N - 1]), Rp2, color='g', fill=False, lw=2)
    for i in range(A1):
        if Nucleus1[i, 6] == 1:
            ax1.plot(Rp1 + Nucleus1[i, 5], Rp1 + Nucleus1[i, 4], 'b.', ms=26, alpha=.6, mew=0, mec='blue')
        if Nucleus1[i, 6] == 0:
            ax1.plot(Rp1 + Nucleus1[i, 5], Rp1 + Nucleus1[i, 4], 'r.', ms=26, alpha=.6, mew=0, mec='red')
    for i in range(A2):
        if Nucleus2[i, 6] == 1:
            ax1.plot(2 * (Rp1 + Rp2) + Nucleus2[i, 5], b[N - 1] + Rp1 + Nucleus2[i, 4], 'g.', ms=26, alpha=.6, mew=0,
                     mec='green')
        if Nucleus2[i, 6] == 0:
            ax1.plot(2 * (Rp1 + Rp2) + Nucleus2[i, 5], b[N - 1] + Rp1 + Nucleus2[i, 4], 'y.', ms=26, alpha=.6, mew=0,
                     mec='yellow')
    ax1.annotate("", xy=(2 * Rp1 + Rp2, Rp1), xycoords='data', xytext=(2 * Rp1, Rp1), textcoords='data',
                 arrowprops=dict(arrowstyle='-|>', connectionstyle="arc"))
    ax1.annotate("", xy=(2 * Rp1, b[N - 1] + Rp1), xycoords='data', xytext=(2 * (Rp1 + Rp2) - Rp2, b[N - 1] + Rp1),
                 textcoords='data', arrowprops=dict(arrowstyle='-|>', connectionstyle="arc"))
    fig1.gca().add_artist(N3)
    fig1.gca().add_artist(N4)
    zed = 2.5 * (Rp1 + Rp2)
    plt.xlim((-zed/2, 1.2*zed))
    plt.ylim((-zed/2, 1.2*zed))
    plt.ylabel('Vertical Position [fm]', fontsize=15)
    plt.xlabel('Horizontal Position [fm]', fontsize=15)
    fig1.set_size_inches(6, 6)
    plt.show()


def PlotResults(b, Npart, Ncoll, Particle1, Particle2, N, Energy, bins=100):
    """Plots number of collisions and participants as a function of impact parameter.
    Shows average trend over data using specified number of bins."""
    bmin = 0
    bmax = max(b)
    E = np.zeros(bins)
    H = np.zeros(bins)
    L = np.zeros(bins)  # ,int64)
    new_b = np.linspace(bmin, bmax, bins)
    newx = np.linspace(bmin, bmax, bins)
    binwidth = (bmax - bmin) / float(bins)
    # Shift by half a bin so the values plot at the right location?
    # If the bins are small enough or the function not too steep, this doesn't matter
    plotx = newx + 0.5 * binwidth
    E2 = np.zeros(bins)
    H2 = np.zeros(bins)
    L2 = np.zeros(bins)  # ,int64)
    newx2 = np.linspace(bmin, bmax, bins)
    binwidth2 = (bmax - bmin) / float(bins)
    plotx2 = newx2 + 0.5 * binwidth2

    # TODO make this all a bit cleaner. I'm currently trying to make a set of averages for the
    # TODO Ncoll and Npart bits. This should make graphing a lot cleaner.
    # As of right now, this doesn't work. Can't figure out why.
    r = len(b)
    b_sort = np.argsort(b)
    means_div = int(r / bins) + int(r % bins != 0)
    new_Ncoll = np.zeros(means_div * bins)
    new_Npart = np.zeros(means_div * bins)
    new_Ncoll[:Ncoll.size] = Ncoll[b_sort]
    new_Npart[:Npart.size] = Npart[b_sort]
    new_Ncoll = new_Ncoll.reshape(bins, means_div)
    new_Ncoll[new_Ncoll == 0.0] = np.nan
    new_Npart = new_Npart.reshape(bins, means_div)
    new_Npart[new_Npart == 0.0] = np.nan
    new_Ncoll_means = np.nanmean(new_Ncoll, axis=1)
    new_Npart_means = np.nanmean(new_Npart, axis=1)
    new_Ncoll_err = np.divide(new_Ncoll_means, np.sqrt(means_div))
    new_Npart_err = np.divide(new_Npart_means, np.sqrt(means_div))

    # Now to get the b values for these means.
    b_means = np.zeros(means_div * bins)
    b_means[:b.size] = b[b_sort]
    b_means = b_means.reshape(bins, means_div)
    b_means_Ncoll = np.multiply(b_means, np.isfinite(np.hstack(new_Ncoll)).reshape(bins, means_div))
    b_means_Npart = np.multiply(b_means, np.isfinite(np.hstack(new_Npart)).reshape(bins, means_div))
    b_means_Ncoll[b_means_Ncoll == 0.0] = np.nan
    b_means_Npart[b_means_Npart == 0.0] = np.nan
    b_means_Ncoll = np.nanmean(b_means_Ncoll, axis=1)
    b_means_Npart = np.nanmean(b_means_Npart, axis=1)

    for i in range(len(b)):
        val = b[i]
        bin = newx.searchsorted(val)
        bin2 = newx2.searchsorted(val)
        L[bin] += 1
        E[bin] += Ncoll[i] ** 2
        H[bin] += Ncoll[i]
        L2[bin2] += 1
        E2[bin2] += Npart[i] ** 2
        H2[bin2] += Npart[i]
    h = H / L
    spr = np.sqrt(E / L - h ** 2)
    err = spr / np.sqrt(L)
    h2 = H2 / L2
    spr2 = np.sqrt(E2 / L2 - h2 ** 2)
    err2 = spr2 / np.sqrt(L2)
    plt.plot(b, Ncoll, "ro", alpha=.9, label='Ncoll')
    plt.plot(b, Npart, "bo", alpha=.5, label='Npart')
    plt.errorbar(b_means_Ncoll, new_Ncoll_means, yerr=new_Ncoll_err, ecolor="purple", elinewidth=2,
                 capsize=2, linewidth=2, color="purple", label='Avg Ncoll')
    plt.errorbar(b_means_Npart, new_Npart_means, yerr=new_Npart_err, ecolor="green", elinewidth=2,
                 capsize=2, linewidth=2, color="green", label='Avg Npart')
    plt.xlim(0, max(b) + 0.5 * binwidth2)
    plt.ylim(0, 1.1 * max(Ncoll))
    plt.legend()
    plt.ylabel('Npart / Ncoll', fontsize=18)
    plt.xlabel('Impact parameter [fm]', fontsize=18)
    plt.title(str(Particle1) + ' + ' + str(Particle2) + '. ' + str(N) + ' iterations. ' + str(Energy) +
              ' center-of-mass energy [GeV].', fontsize=18)
    plt.show()
