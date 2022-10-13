#
# \Calibrate nSigmaProton
#
# \author Skipper Kagamaster
# \date 10/8/2022
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
# /

"""
The purpose of this code is to calibrate nSigmaProton. There are two methods
used to accomplish this:
1. Using a smoothed function.
This method leverages a Savitsky-Golay folter to smooth the
function, then take the shape of the second derivative at
inflection points to characterise maxima. The one closest
to 0 is taken as the nSigmaProton calibrated mean value.
2. Multiple Gaussian fit
This method takes a more physics based approach and fits the
spectra with 2 Gaussian functions, taking the mean of the
one closest to 0 as the nSigmaProton calibrated mean value.
The fitting will try first a 2-Gaussian fit, then a single
Gaussian, then fall back to a Savitsky-Golay filter (and will
ultimately just leave the mean peak at 0 if all else fails).
"""

import numpy as np
import uproot as up
import awkward as ak
import os
import functions as fn
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter as sgf
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema as arex
from scipy.optimize import curve_fit


ROOT = False

if ROOT is True:
    os.chdir(r'D:\14GeV\Thesis\After_Qa_Picos')
    pdf_pages = PdfPages(r'D:\14GeV\Thesis\nSigmaProton.pdf')
else:
    os.chdir(r'D:\14GeV\Thesis\PythonArrays\nSigmaProtonCal')
    pdf_pages = PdfPages(r'D:\14GeV\Thesis\nSigmaPython.pdf')


def gaussian(x, A, mu, sigma):
    return A*(1/(sigma*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-mu)/sigma)**2)))


# NOTE: You can easily use this template to make fit functions for more than 2 Gaussians.
def double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
    return A1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-mu1)/sigma1)**2))) + \
            A2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-mu2)/sigma2)**2)))


nSigLabels = ['hNSigmaProton_0', 'hNSigmaProton_1', 'hNSigmaProton_2', 'hNSigmaProton_3',
              'hNSigmaProton_4', 'hNSigmaProton_5', 'hNSigmaProton_6', 'hNSigmaProton_7',
              'hNSigmaProton_8', 'hNSigmaProton_9', 'hNSigmaProton_10']
pt_vals = [r'$p_T$ = 0.1-0.2 $\frac{GeV}{c}$', r'$p_T$ = 0.2-0.3 $\frac{GeV}{c}$',
           r'$p_T$ = 0.3-0.4 $\frac{GeV}{c}$', r'$p_T$ = 0.4-0.5 $\frac{GeV}{c}$',
           r'$p_T$ = 0.5-0.6 $\frac{GeV}{c}$', r'$p_T$ = 0.6-0.7 $\frac{GeV}{c}$',
           r'$p_T$ = 0.7-0.8 $\frac{GeV}{c}$', r'$p_T$ = 0.8-0.9 $\frac{GeV}{c}$',
           r'$p_T$ = 0.9-1.0 $\frac{GeV}{c}$', r'$p_T$ = 1.0-1.1 $\frac{GeV}{c}$',
           r'$p_T$ = 1.1-1.2 $\frac{GeV}{c}$']
files = os.listdir()
runs = []
nSigCal = []
# nSigCalGauss = []
count = 0
print("Working on file:")
for k in range(len(files)):
    if (k+1) % 100 == 0:
        print(k+1)
    if ROOT is True:
        data = up.open(files[k])
        # Avoid empty files.
        if np.mean(data['hNSigmaProton_1'].to_numpy()[0]) == 0:
            continue
        runs.append(files[k][4:12])
        nSigs = []
        xaxes = []
        for j in nSigLabels:
            nSigs.append(data[j].to_numpy()[0])
            xaxes.append(data[j].to_numpy()[1][:-1])
    else:
        nSigs = np.load(files[k], allow_pickle=True)
        xaxes = np.linspace(-10, 10, 400)
        xaxes = np.tile(xaxes, (len(nSigs), 1))
        runs.append(files[k][:8])
    nSigCal.append([])
    # And here's the double Gauss fit.
    true_mu = 0
    fig, ax = plt.subplots(3, 4, figsize=(16, 9), constrained_layout=True)
    for i in range(11):  # We can go up to 13, but convention is just the first 11.
        a = int(i/4)
        b = i % 4
        try:
            popt_2gauss, pcov_2gauss = curve_fit(double_gaussian, xaxes[i], nSigs[i],
                                                 p0=[np.max(nSigs[i]), 0, 1, np.max(nSigs[i]), -5, 1],
                                                 maxfev=int(1e5))
            perr_2gauss = np.sqrt(np.diag(pcov_2gauss))
            pars_1 = popt_2gauss[0:3]
            pars_2 = popt_2gauss[3:6]
            gauss_peak_1 = gaussian(xaxes[i], *pars_1)
            gauss_peak_2 = gaussian(xaxes[i], *pars_2)
            # Now take the mean closest to 0.
            true_mu = pars_1[1]
            if abs(true_mu) > 2.5:
                print("Mu value out of range!")
                raise ValueError
            ax[a, b].plot(xaxes[i], nSigs[i], color='black', label='raw')
            ax[a, b].plot(xaxes[i], double_gaussian(xaxes[i], *popt_2gauss), lw=3, color='orange', label="Gaussian Fit")
            ax[a, b].plot(xaxes[i], gauss_peak_1, "g")
            ax[a, b].fill_between(xaxes[i], gauss_peak_1.min(), gauss_peak_1, facecolor="green", alpha=0.5)
            ax[a, b].plot(xaxes[i], gauss_peak_2, "y")
            ax[a, b].fill_between(xaxes[i], gauss_peak_2.min(), gauss_peak_2, facecolor="yellow", alpha=0.5)
        except Exception as e:  # Fall back to fitting with a single Gaussian if the above fails.
            print("File", files[k], "p_T", pt_vals[i], '\n',
                  "Too close for missles; switching to guns.")
            try:
                popt_gauss, pcov_gauss = curve_fit(gaussian, xaxes[i], nSigs[i],
                                                   p0=[np.max(nSigs[i]), 0, 1],
                                                   maxfev=int(1e5))
                true_mu = popt_gauss[1]
                if abs(true_mu) > 2.5:
                    print("Mu value out of range ... again!")
                    raise ValueError
                gauss_peak_1 = gaussian(xaxes[i], *popt_gauss)
                ax[a, b].plot(xaxes[i], nSigs[i], color='black', label='raw')
                ax[a, b].plot(xaxes[i], gaussian(xaxes[i], *popt_gauss), lw=3, color='orange', label="Gaussian Fit")
                ax[a, b].plot(xaxes[i], gauss_peak_1, "g")
                ax[a, b].fill_between(xaxes[i], gauss_peak_1.min(), gauss_peak_1, facecolor="green", alpha=0.5)
            except Exception as e:  # Fall back to SavGol as a last resort.
                try:
                    print("It's now a smooth criminal.")
                    sg = sgf(nSigs[i], 201, 6)
                    maxes = np.asarray(xaxes[i][np.hstack(np.asarray(arex(sg, np.greater, order=10)))])
                    # The selection is based on the plots; you'll have to
                    # play around with this based on how many have failed
                    # Gaussian fits (look them over manually).
                    if i == 0:
                        maxes = maxes[maxes < 0]
                        if maxes.size > 1:
                            true_mu = maxes[-1]  # Pick the first one less than 0.
                        else:
                            true_mu = maxes
                    else:
                        index = np.argmin(np.absolute(maxes))
                        if index.size > 1:
                            index = index[0]  # Just in case 2 mins are equally close to 0.
                        true_mu = maxes[index]
                    if abs(true_mu) > 2.5:
                        print("Mu value out of range ... still!")
                        print("But I'm keeping it. You can't stop me.")
                    ax[a, b].plot(xaxes[i], nSigs[i], color='black', label='raw')
                    ax[a, b].plot(xaxes[i], sg, lw=3, color='pink', label="Smoothed")
                    for j in maxes:
                        ax[a, b].axvline(j, color='red', label=r'$\frac{d^2f}{dx^2}$ max')
                    ax[a, b].axvline(true_mu, color='blue', lw=3, label='p peak')
                    ax[a, b].set_xlabel(r'$n\sigma_{p}$', fontsize=12, loc='right')
                    ax[a, b].set_ylabel('N', fontsize=12, loc='top')
                    ax[a, b].set_title(pt_vals[i], fontsize=15)
                    ax[-1, -1].plot(1, c='r', lw=4, label=r'$\frac{d^2f}{dx^2}$ max')
                    ax[-1, -1].plot(1, c='pink', lw=4, label=r'SavGol')
                except Exception as e:
                    true_mu = 0  # This is just giving up.
        nSigCal[count].append(true_mu)
        ax[a, b].axvline(true_mu, color='blue', lw=3, label='p peak')
        ax[a, b].set_xlabel(r'$n\sigma_{p}$', fontsize=12, loc='right')
        ax[a, b].set_ylabel('N', fontsize=12, loc='top')
        ax[a, b].set_title(pt_vals[i], fontsize=15)
    ax[-1, -1].set_axis_off()
    ax[-1, -1].plot(1, c='yellow', lw=4, label=r'1st Gaussian')
    ax[-1, -1].plot(1, c='green', lw=4, label=r'2nd Gaussian')
    ax[-1, -1].plot(1, c='orange', lw=4, label=r'2 Gaussian Fit')
    ax[-1, -1].plot(1, c='black', lw=4, label=r'Spectra')
    ax[-1, -1].plot(1, c='blue', lw=4, label=r'$n\sigma_p$ $\mu$')
    ax[-1, -1].legend(fontsize=20, loc='center')
    fig.suptitle(str(runs[count]), fontsize=20)
    pdf_pages.savefig(fig)
    plt.close()
    count += 1
pdf_pages.close()
runs = np.asarray(runs).astype('int')
print(runs.T)
nSigCal = np.asarray(nSigCal)
# nSigCalGauss = np.asarray(nSigCalGauss)
np.savetxt(r'D:\14GeV\Thesis\nSigmaCal.txt', nSigCal, fmt='%f', delimiter=',', newline='},{')
# np.savetxt(r'D:\14GeV\Thesis\nSigmaProton_2g.txt', nSigCalGauss, fmt='%f', delimiter=',', newline='},{')
np.savetxt(r'D:\14GeV\Thesis\runs.txt', runs.T, fmt='%d', delimiter=',', newline=',')
np.save(r'D:\14GeV\Thesis\PythonArrays\nSigmaCal.npy', nSigCal)
# np.save(r'D:\14GeV\Thesis\PythonArrays\nSigmaProton_2g.npy', nSigCalGauss)
np.save(r'D:\14GeV\Thesis\PythonArrays\runs.npy', runs.T)
