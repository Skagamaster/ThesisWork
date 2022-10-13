"""The purpose of this code is to analyse 200 GeV EPD data for
calibration (all in Python). I made spectra histograms in the
macro SmoothADCTest.py, and will load and analyse them here.
This could all be a single macro, but loading the data and
creating the spectra histograms can be pretty slow due to their
size, so here we are.

You will have to have uproot and awkward installed for this to
run; versions as of this writing are uproot 4 and awkward 1.0.

-Skipper Kagamaster
skk317@lehigh.edu
9/26/2022
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import moyal
from scipy.signal import savgol_filter as sgf
from scipy.signal import argrelextrema as arex


# Here's a Gaussian function.
def gaussian(x_arr, A, mu, sigma):
    return A * (1 / (sigma * (np.sqrt(2 * np.pi)))) * (np.exp((-1.0 / 2.0) * (((x_arr - mu) / sigma) ** 2)))


# Here's a double Gaussian function.
# NOTE: You can easily use this template to make fit functions for more than 2 Gaussians.
def double_gaussian(x_arr, A1, mu1, sigma1, A2, mu2, sigma2):
    return A1 * (1 / (sigma1 * (np.sqrt(2 * np.pi)))) * (np.exp((-1.0 / 2.0) * (((x_arr - mu1) / sigma1) ** 2))) + \
           A2 * (1 / (sigma2 * (np.sqrt(2 * np.pi)))) * (np.exp((-1.0 / 2.0) * (((x_arr - mu2) / sigma2) ** 2)))


# Here's a scaled moyal.
def mult_moyal_scaled(x_arr, loc, *wid_scale, nMIPs=3):
    func = wid_scale[0] * moyal.pdf(x_arr, loc, wid_scale[1])
    for i in range(1, nMIPs):
        func += wid_scale[i] * moyal.pdf(x_arr, (i+1) * loc, wid_scale[i+1])
    return func


# More lembas ...
def moyal_scaled(x_arr, loc, wid, scale):
    return scale * moyal.pdf(x_arr, loc, wid)


# Now let's make some ridiculous graphs for offline analysis. Include a metric for doing
# a SavGol filter and finding extrema (could also do it with the 2nd derivative of the
# SavGol, but this way we can more easily define min/max).
def make_plots(x_arr, spectra, cut_off, savgol=True, save_dir=r'C:\200', names=(191, 192, 193)):
    os.chdir(save_dir)
    for i in range(3):
        file_name = '200_GeV_spectra_day_{}.pdf'.format(names[i])
        array_name = '200_GeV_spectra_day_{}.npy'.format(names[i])
        arr = np.zeros((2, 12, 31))
        adc_smooth = None
        if savgol is True:
            file_name = '200_GeV_spectra_savgol_day_{}.pdf'.format(names[i])
            adc_smooth = sgf(sgf(adc_spectra, 41, 3), 81, 3)
        with PdfPages(file_name) as pdf:
            for j in range(2):
                for k in range(12):
                    fig, ax = plt.subplots(4, 8, figsize=(16, 9), constrained_layout=True)
                    plt.suptitle("EW:" + str(j) + ", PP:" + str(k + 1), fontsize=20)
                    for m in range(4):
                        for n in range(8):
                            r = m * 8 + n
                            if r >= 31:
                                ax[m, n].set_axis_off()
                                continue
                            ax[m, n].plot(x[:cutoff], adc_spectra[i][j][k][r][:cutoff], c='blue')
                            if savgol is True:
                                ax[m, n].plot(x[:cutoff], adc_smooth[i][j][k][r][:cutoff], c='orange')
                                y = arex(adc_smooth[i][j][k][r], np.greater, order=20)
                                y = np.hstack(y)
                                scale_guess = adc_smooth[i][j][k][r][y[1]]
                                ax[m, n].axvline(y[1], c='r')
                                arr[j][k][r] = y[1]
                                """
                                popt_moyal, pcov_moyal = curve_fit(mult_moyal_scaled, x[50:1500],
                                                                   adc_spectra[i][j][k][r][50:1500],
                                                                   p0=[y[1], 1, scale_guess,
                                                                                  scale_guess/2,
                                                                                  scale_guess/3],
                                                                   bounds=[[y[1] - 50, 0, 0, 0, 0],
                                                                           [y[1] + 150, 200, np.inf,
                                                                            np.inf, np.inf]],
                                                                   maxfev=int(1e5))
                                perr_2gauss = np.sqrt(np.diag(pcov_moyal))
                                print(y[1], 40, 165 * scale_guess)
                                print(popt_moyal)
                                print(pcov_moyal)
                                plt.close()
                                plt.plot(x[50:], adc_spectra[i][j][k][r][50:], c='blue')
                                plt.plot(mult_moyal_scaled(x[50:], *popt_moyal), c='r')
                                color = ['orange', 'purple', 'green']
                                for o in range(3):
                                    plt.plot(popt_moyal[2+o] * moyal.pdf(x[15:], (o + 1) * popt_moyal[0],
                                                                         (o + 1) * popt_moyal[1]), c=color[o])
                                    print(np.sum(popt_moyal[2+o] * moyal.pdf(x[15:], (o + 1) * popt_moyal[0],
                                                                             (o + 1) * popt_moyal[1])))
                                # plt.yscale('log')
                                plt.show()
                                """
                            else:
                                y = arex(adc_spectra[i][j][k][r], np.greater, order=40)
                                y = np.hstack(y)
                                y = y[y <= cutoff]
                                if len(y) > 1:
                                    arr[j][k][r] = y[0]
                                else:
                                    print(i, j, k, r, "had a y len of 0.")
                                for q in range(len(y)):
                                    ax[m, n].axvline(y[q], c='r')
                            ax[m, n].set_title("TT " + str(r + 1), fontsize=15)
                            ax[m, n].set_xlabel("ADC", fontsize=10)
                            ax[m, n].set_yscale('log')
                    pdf.savefig()
                    plt.close()
        np.save(array_name, arr)

# Load the spectra data.
save_dir = r'C:\200'
os.chdir(save_dir)
nums = (191, 192, 193)
adc_spectra = []
for i in range(3):
    adc_spectra.append(np.load('cal_vals_{}.npy'.format(nums[i]), allow_pickle=True))

# This is our x-axis (from the original histogram).
x = np.linspace(0, 1999, 2000)
cutoff = 400  # To truncate the graph output so it's meaningful.

# And let's generate the plots.
nums = [191, 192, 193]
make_plots(x, adc_spectra, cutoff, savgol=True)
print("And you're gone; you're a ghost.")
