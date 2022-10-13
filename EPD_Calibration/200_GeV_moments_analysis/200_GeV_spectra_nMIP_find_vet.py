# Built to see where peaks were found from 200_gEv_review_ADC_spectra.py.

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import moyal
from scipy.signal import savgol_filter as sgf
from scipy.signal import argrelextrema as arex

os.chdir(r'C:\200\cal_vals')
day_set = [191, 192, 193]

spectra = []
maxes = []

# This is our x-axis (from the original histogram).
x = np.linspace(0, 1999, 2000)
cutoff = 400  # To truncate the graph output so it's meaningful.

# Loading data, but not the SavGol at this time.
for i in day_set:
    spectra.append(np.load('cal_vals_{}.npy'.format(i), allow_pickle=True))
    maxes.append(np.load('200_GeV_spectra_fix_day_{}.npy'.format(i), allow_pickle=True))

"""
maxes[0][0][1][16] = 76.0
maxes[0][0][10][19] = 72.0
maxes[0][0][10][21] = 70.0
maxes[0][1][7][17] = 101.0

maxes[1][0][1][16] = 75.0
maxes[1][0][10][19] = 72.0
maxes[1][0][10][21] = 71.0
maxes[1][1][7][17] = 96.0

maxes[2][0][10][19] = 72.0
maxes[2][0][10][21] = 69.0

for i in range(3):
    np.save('200_GeV_spectra_fix_day_{}.npy'.format(day_set[i]), maxes[i])
print("Boom. Roasted.")
exit()
"""

for i in range(3):
    file_name = '200_GeV_max_loc_day_{}.pdf'.format(day_set[i])
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
                        ax[m, n].plot(x[:cutoff], spectra[i][j][k][r][:cutoff], c='blue')
                        ax[m, n].axvline(maxes[i][j][k][r], c='r')
                        ax[m, n].set_title("TT " + str(r + 1), fontsize=15)
                        ax[m, n].set_xlabel("ADC", fontsize=10)
                        ax[m, n].set_yscale('log')
                pdf.savefig()
                plt.close()
