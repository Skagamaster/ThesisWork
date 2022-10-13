import numpy as np
import pandas as pd
import uproot as up
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as sgf
from scipy.signal import argrelextrema as arex
from scipy.stats import moyal
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
import pico_reader as pr
import glob


# This function takes an index and returns the EWxPPyTTz information.
def EPDTile(pos=123):
    ew = int(pos / 372)
    pp = int((pos - 372 * ew) / 31) + 1
    tt = (pos % 31) + 1
    return [ew, pp, tt]


def epd_tile(a):
    ew = (a / 372).astype(int)
    pp = ((a - 372 * ew) / 31 + 1).astype(int)
    tt = ((a % 31) + 1).astype(int)
    return ew, pp, tt


# Number of curves to consider in Moyal fit.
nMIPs = int(2)

"""
I'm restarting this code using the SciPy methodology to just get a fit on a known tile.
Let's see how we get on.
"""
list_200 = [r'F:\AuAu200\191', r'F:\AuAu200\192', r'F:\AuAu200\193']
nums = [191, 192, 193]
for k in range(len(list_200)):
    print("***********************************************")
    print("Now serving day", nums[k])
    os.chdir(list_200[k])
    picofile_list = glob.glob('*.root')
    epd_array = np.zeros((2, 12, 31, 2000))
    print("Working on file:")
    counter = 0
    for file in picofile_list:
        if counter % 50 == 0:
            print(counter+1, "of", len(picofile_list))
        counter += 1
        try:
            pico = pr.PicoDST()
            pico.import_data(file)
            for i in range(744):
                count, bins = np.histogram(pico.epd_tiles[:, i], bins=2000, range=(0, 2000))
                epd_array[EPDTile(i)[0]][EPDTile(i)[1]-1][EPDTile(i)[2]-1] += count
        except Exception as e:
            continue
    os.chdir(r'C:\200')
    np.save('cal_vals_{}.npy'.format(nums[k]), epd_array)
print("All done!")
exit()

# Grab the data to be worked on.
os.chdir(r'C:\PhysicsProcessing\7.7GeV\Days')
daySet = np.asarray((31, 32, 33, 34, 35, 36, 37))
for h in daySet:
    data = up.open('0{}.root'.format(h))
    # This section of code smooths out the ADC distributions (helpful for
    # individual runs or low statistics), then finds the local minima and
    # maxima of those smoothed distributions.
    r = int(len(data.keys()))
    ADC = []
    for i in data.keys():
        ADC.append(data[i].values())
    ADC = np.asarray(ADC)
    x = np.linspace(0, len(ADC[0])-1, len(ADC[0]))
    y = []
    exMin = []
    exMax = []

    # Why is this here?
    savLen = int(len(ADC[0]))
    if savLen % 2 == 0:
        savLen = savLen-1
    savPol = int(savLen*0.01)

    # Triple sgf filter; find the extrema.
    adc_smooth = sgf(sgf(sgf(ADC, 141, 5), 71, 3), 51, 2)
    y = arex(adc_smooth, np.greater, order=15)
    print(y)
    plt.plot(ADC[1], lw=3)
    plt.plot(adc_smooth[1], lw=2)
    plt.show()

    for i in range(r):
        placeholder = sgf(ADC[i], 141, 5)  # initial smoothing
        placeholder = sgf(placeholder, 71, 3)  # intermediate smoothing
        placeholder = sgf(placeholder, 51, 2)  # final smoothing
        y.append(placeholder)
        if len(arex(y[i], np.greater, order=15)[0]) == 0:
            exMax.append([1000])
        else:
            exMax.append(np.asarray(
                arex(y[i], np.greater, order=15)).flatten())
        exMin.append(np.asarray(arex(y[i], np.less, order=15)).flatten())
    y = np.asarray(y)
    exMin = np.asarray(exMin)
    exMax = np.asarray(exMax)

    fitStart = []
    startVal = []
    weightGuess = []
    guess = []
    boundsInit = []
    boundsEnd = []
    for q in range(int(len(ADC))):
        # Check for an empty tile
        if exMin[q].size == 0:
            print(EPDTile(q))
            fitStart.append(0)
            startVal.append(0)
            weightGuess.append(0)
            guess.append([nMIPs, 0, 0, 0])
            boundsInit.append([nMIPs, 0, 0, 0])
            boundsEnd.append([nMIPs+1, 0, 0, 0.3*startVal[q]])
            for i in range(1, nMIPs):
                l = i+1
                guess[q].extend(
                    [weightGuess[q]/l, l*startVal[q], 0.2*startVal[q]])
                boundsInit[q].extend([0, fitStart[q]*l*0.75, 0])
                boundsEnd[q].extend([np.inf, np.inf, np.inf])
            continue
        first = exMin[q][0]
        if first < 15:
            first = exMin[q][1]
        MIPguess = exMax[q][np.where(exMax[q] > first+20)[0][0]]
        starter = int(first+abs(first-MIPguess)/2)
        fitStart.append(starter)
        startVal.append(MIPguess)
        weightGuess.append(np.max(y[q]))
        guess.append([nMIPs, weightGuess[q], startVal[q], 0.2*startVal[q]])
        boundsInit.append([nMIPs, 0, fitStart[q], 0])
        boundsEnd.append([nMIPs+1, np.inf, np.inf, 0.3*startVal[q]])
        for i in range(1, nMIPs):
            l = i+1
            guess[q].extend([weightGuess[q]/l, l*startVal[q], 0.2*startVal[q]])
            boundsInit[q].extend([0, fitStart[q]*l*0.75, 0])
            boundsEnd[q].extend([np.inf, np.inf, np.inf])

    fitStart = np.asarray(fitStart)
    startVal = np.asarray(startVal)
    arr = np.asarray((fitStart, startVal)).T.astype(int)
    df = pd.DataFrame(arr)
    np.savetxt(r'C:\PhysicsProcessing\7.7GeV\Day{}starts.txt'.format(
        h), arr, fmt='%d')

    # Now to graph the found fits.
    print("Working on ...")
    with PdfPages(r'C:\PhysicsProcessing\7.7GeV\nMIP_plots{}.pdf'.format(h)) as export_pdf:
        for m in range(2):
            for n in range(12):
                print(f"EW{EPDTile(m*372+n*31)[0]}PP{EPDTile(m*372+n*31)[1]}")
                fig, ax = plt.subplots(6, 6, figsize=(16, 10))
                for s in range(31):
                    i = int(s/6)
                    j = s % 6
                    l = m*372+n*31+s
                    ax[i, j].plot(x, ADC[l], c='black',
                                  lw=1, label="ADC Spectra")
                    ax[i, j].plot(x, y[l], c='r', lw=1,
                                  label="ADC Smoothed")
                    color = ['gray', 'red', 'blue',
                             'purple', 'green', 'yellow']
                    for k in exMin[l]:
                        ax[i, j].axvline(
                            x=k, c='pink', lw=1, label="Minima")
                    for k in exMax[l]:
                        ax[i, j].axvline(x=k, c='g', lw=1, label="Maxima")
                    ax[i, j].axvline(
                        x=fitStart[l], c='blue', label="FitStart")
                    ax[i, j].axvline(
                        x=startVal[l], c='orange', label="StartVal")
                    ax[i, j].set_xlim(0, 250)
                    ax[i, j].set_ylim(0, y[l][startVal[l]]*2)
                    ax[i, j].set_xlabel("ADC", fontsize=10)
                    ax[i, j].set_ylabel("Counts", fontsize=10)
                    ax[i, j].set_title(
                        f"EW{EPDTile(l)[0]}PP{EPDTile(l)[1]}TT{EPDTile(l)[2]}", fontsize=10)

                legendMaster = ["ADC Spectra", "ADC Smoothed",
                                "Maxima", "Minima", "FitStart", "MIPGuess"]
                color = ["black", "red", "green", "pink", "blue", "orange"]
                for t in range(int(len(color))):
                    ax[5, 1].plot(0)
                ax[5, 1].set_axis_off()
                leg = ax[5, 1].legend(legendMaster, fontsize=10)
                count = 0
                for line in leg.get_lines():
                    line.set_linewidth(2)
                    line.set_color(color[count])
                    count += 1
                ax[5, 3].text(
                    0, 0, f"EW{EPDTile(m*372+n*31)[0]}PP{EPDTile(m*372+n*31)[1]}", size=40)
                ax[5, 2].set_axis_off()
                ax[5, 3].set_axis_off()
                ax[5, 4].set_axis_off()
                ax[5, 5].set_axis_off()
                plt.tight_layout()
                export_pdf.savefig()
                plt.close()
    print("Boom; roasted.")
