import numpy as np
import uproot as up
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as sgf
from scipy.signal import argrelextrema as arex
from scipy.stats import moyal
from scipy.optimize import curve_fit

# Number of curves to consider in Moyal fit.
nMIPs = int(2)

# Grab the data to be worked on.
os.chdir(r'C:\PhysicsProcessing\9.2GeV\Days')
# data = up.open('20138043.root')
data = up.open("21030.root")

# This function takes an index and returns the EWxPPyTTz information.


def EPDTile(pos=123):
    ew = int(pos/372)
    pp = int((pos-372*ew)/31)
    tt = (pos % 31) + 1
    pp = pp+1
    return [ew, pp, tt]


# This section of code smooths out the ADC distributions (helpful for
# individual runs or low statistics), then finds the local minima and
# maxima of those smoothed distributions.
r = int(len(data.keys()))
ADC = []
for i in range(r):
    ADC.append(np.asarray(data[data.keys()[i]][:]))
ADC = np.asarray(ADC)
x = np.linspace(0, len(ADC[0])-1, len(ADC[0]))
y = []
exMin = []
exMax = []
savLen = int(len(ADC[0]))
if savLen % 2 == 0:
    savLen = savLen-1
savPol = int(savLen*0.01)
for i in range(r):
    placeholder = sgf(ADC[i], 101, 7)  # initial smoothing
    placeholder = sgf(placeholder, 301, 3)  # final smoothing
    y.append(placeholder)
    if len(arex(y[i], np.greater, order=15)[0]) == 0:
        exMax.append([1000])
    else:
        exMax.append(np.asarray(arex(y[i], np.greater, order=15)).flatten())
    exMin.append(np.asarray(arex(y[i], np.less, order=15)).flatten())
y = np.asarray(y)
exMin = np.asarray(exMin)
exMax = np.asarray(exMax)

# This is the function we'll be fitting with. I used Moyal curves because Python
# doesn't have Landau curves (as of this coding); moyal is a decent approximation.


def multMoyal(x, nMIPs, *args):
    moy = 0
    for i in range(int(nMIPs)):
        l = 3*i
        moy = moy + args[l]*moyal.pdf(x, args[l+1], args[l+2])
    return moy


# Let's set some initial parameters for the fit. The main ones are going to be the weight,
# the 1st MIP MPV guess, and the width. Initial guesses all default to 1, so really anything
# we put here is better than the defaults. WID/MPV should = ~0.2 for the EPD.
fitStart = []
startVal = []
weightGuess = []
guess = []
boundsInit = []
boundsEnd = []
for q in range(int(len(ADC))):
    MIPguess = exMax[q][np.where(exMax[q] > exMin[q][0])[0][0]]-10
    starter = int(exMin[q][0]+abs(exMin[q][0]-MIPguess)/2)
    fitStart.append(starter)
    startVal.append(MIPguess)
    weightGuess.append(y[q][MIPguess])
    guess.append([nMIPs, weightGuess[q], startVal[q], 0.2*startVal[q]])
    boundsInit.append([nMIPs, 0, fitStart[q], 0])
    boundsEnd.append([nMIPs+1, np.inf, np.inf, 0.3*startVal[q]])
    for i in range(1, nMIPs):
        l = i+1
        guess[q].extend([weightGuess[q]/l, l*startVal[q], 0.2*startVal[q]])
        boundsInit[q].extend([0, fitStart[q]*l*0.75, 0])
        boundsEnd[q].extend([np.inf, np.inf, np.inf])

# Now for the fit itself. y1 will be the fit, c will hold the information on
# individual moyal ccurves, and MPV will be the 1st MIP MPV and error.
y1 = []
c = []
MPV = []
print("Working on tile:")
for q in range(11):
    print(EPDTile(q))
    c.append([])
    try:
        param, param_cov = curve_fit(
            multMoyal, x[fitStart[q]:], y[q][fitStart[q]:],
            maxfev=5000, p0=guess[q],
            bounds=(boundsInit[q], boundsEnd[q]))
        MPV.append([param[2], param[3]])
        y1.append(multMoyal(x, *param))
        for i in range(nMIPs):
            l = i*3+1
            c[q].append(param[l]*moyal.pdf(x, param[l+1], param[l+2]))
    except (RuntimeError, ValueError):
        MPV.append([0, 0])
        y1.append(moyal.pdf(x, 0, 0))
        for i in range(nMIPs):
            c[q].append(moyal.pdf(x, 0, 0))
        print(f"Fit failed for {EPDTile(q)}.")

# Now to graph the found fits. This is currently for a single supersector.
plotnum = int(3)
fig, ax = plt.subplots(4, 3, figsize=(16, 10))
for i in range(4):
    for j in range(3):
        l = int(i*3+j)
        if l < 11:
            ax[i, j].plot(x, ADC[l], c='black', lw=1, label="ADC Spectra")
            ax[i, j].plot(x, y[l], c='r', lw=1, label="ADC Smoothed")
            ax[i, j].plot(x, y1[l], c='b', lw=0, label="Fit")
            color = ['gray', 'red', 'blue', 'purple', 'green', 'yellow']
            for k in range(nMIPs):
                ax[i, j].fill_between(
                    x, c[l][k], color=color[k], alpha=0.5, label=f"{k+1} MIP")
            ax[i, j].axvline(x=MPV[l][0], c='g', lw=1, label="1st MIP MPV")
            ax[i, j].axvline(x=fitStart[l], c='pink', lw=1, label="FitStart")
            ax[i, j].set_xlim(0, 500)
            ax[i, j].set_ylim(0, 3*weightGuess[l])
            ax[i, j].set_xlabel("ADC", fontsize=10)
            ax[i, j].set_ylabel("Counts", fontsize=10)
            ax[i, j].set_title(
                f"EW{EPDTile(l)[0]}PP{EPDTile(l)[1]}TT{EPDTile(l)[2]}", fontsize=10)

legendMaster = ["ADC Spectra", "ADC Smoothed", "Fit", "1st MIP MPV"]
color = ["black", "red", "blue", "green", 'gray',
         'red', 'blue', 'purple', 'green', 'yellow']
for i in range(nMIPs):
    legendMaster.append(f"{i+1} MIP")
# for i in range(nMIPs+4):
#    ax[plotnum+1, 1].plot(0)
ax[3, 2].plot(0)
ax[3, 2].set_axis_off()
#ax[plotnum+1, 2].set_axis_off()
#ax[plotnum+1, 3].set_axis_off()
#ax[plotnum+1, 4].set_axis_off()
#ax[plotnum+1, 4].text(0, 0, f"EW{EPDTile(0)[0]}PP{EPDTile(0)[1]}", size=30)
leg = ax[3, 2].legend(legendMaster, fontsize=10)
count = 0
for line in leg.get_lines():
    line.set_linewidth(2)
    line.set_color(color[count])
    count += 1

plt.tight_layout()
plt.show()
