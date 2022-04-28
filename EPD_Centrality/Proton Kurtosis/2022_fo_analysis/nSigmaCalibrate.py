import numpy as np
import uproot as up
import awkward as ak
import os
import functions as fn
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter as sgf
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema as arex


os.chdir(r'D:\14GeV\Thesis\After_Qa_Picos')
pdf_pages = PdfPages(r'D:\14GeV\Thesis\nSigmaProton.pdf')

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
count = 0
print("Working on file:")
for k in range(len(files)):
    data = up.open(files[k])
    if (k+1) % 100 == 0:
        print(k+1)
    if np.mean(data['hNSigmaProton_1'].to_numpy()[0]) == 0:
        continue
    runs.append(files[k][4:12])
    nSigCal.append([])
    nSigs = []
    xaxes = []
    for j in nSigLabels:
        nSigs.append(data[j].to_numpy()[0])
        xaxes.append(data[j].to_numpy()[1][:-1])
    fig, ax = plt.subplots(3, 4, figsize=(16, 9), constrained_layout=True)
    for i in range(len(nSigs)):
        a = int(i/4)
        b = i % 4
        sg = sgf(nSigs[i], 201, 6)
        maxes = np.asarray(
            xaxes[i][np.hstack(np.asarray(arex(sg, np.greater, order=10)))])
        # Now take the max closest to 0.
        index = np.argmin(np.absolute(maxes))
        if index.size > 1:
            print(runs[count])
            index = index[0]  # Just in case 2 mins are equally close to 0.
        nSigCal[count].append(maxes[index])
        ax[a, b].plot(xaxes[i], nSigs[i], color='black', label='raw')
        ax[a, b].plot(xaxes[i], sg, lw=3, color='orange', label="Smoothed")
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
    fig.suptitle(runs[count], fontsize=20)
    # fig = fn.savGol(nSigs, xaxes, run=runs[i])
    pdf_pages.savefig(fig)
    plt.close()
    count += 1
pdf_pages.close()
runs = np.asarray(runs).astype('int')
print(runs.T)
nSigCal = np.asarray(nSigCal)
np.savetxt(r'D:\14GeV\Thesis\nSigmaProton.txt', nSigCal, fmt='%f', delimiter=',', newline='},{')
np.savetxt(r'D:\14GeV\Thesis\runs.txt', runs.T, fmt='%d', delimiter=',', newline=',')
