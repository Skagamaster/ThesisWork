import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import simplejson
import uproot as up
from matplotlib.backends.backend_pdf import PdfPages


def EPDTile(tile=int(123)):
    if tile < 372:
        ew = 0
    else:
        ew = 1
    pp = int((tile - int(tile/372)*372)/31) + 1
    tt = tile % 31 + 1
    return [ew, pp, tt]


# os.chdir(r"D:\14GeV")
os.chdir(r'C:\PhysicsProcessing\7.7_Ped_check')

print("Importing data ...")

names = np.load("pednames.npy", allow_pickle=True)
peds = np.load("pedestals.npy", allow_pickle=True)

investigate = []
deltas = []
count = 0
for i in peds:
    investigate.append([])
    investigate[count].append(names[count])
    for j in range(int(len(i))):
        if j > 9:
            investigate[count].append(EPDTile(j))
    if count != 0:
        deltas.append(peds[count-1]-i)
    count += 1

lister = []
count = 1
for i in deltas:
    lister.append([])
    lister[count-1].append(names[count])
    count1 = 0
    for j in i:
        if abs(j) > 15:
            lister[count-1].append(EPDTile(count1))
        count1 += 1
    count += 1

f = open('pedestals_to_review_155-156.txt', 'w')
simplejson.dump(lister, f)

print("Building ADC spectra ...")

Full_ADC_Spectra = []
# os.chdir(r"D:\14GeV\Day137\Runs")
os.chdir(r"C:\PhysicsProcessing\7.7_Ped_check\Day157")
for i in os.listdir():
    if Path(i).suffix != '.root':
        continue
    data = up.open(i)
    r = int(len(data.keys()))
    ADC = []
    for j in range(r):
        ADC.append(np.asarray(data[data.keys()[j]][:]))
    ADC = np.asarray(ADC)
    x = np.linspace(0, len(ADC[0])-1, len(ADC[0]))
    Full_ADC_Spectra.append(ADC)
# os.chdir(r"D:\14GeV\Day138\Runs")
os.chdir(r"C:\PhysicsProcessing\7.7_Ped_check\Day158")
for i in os.listdir():
    if Path(i).suffix != '.root':
        continue
    data = up.open(i)
    r = int(len(data.keys()))
    ADC = []
    for j in range(r):
        ADC.append(np.asarray(data[data.keys()[j]][:]))
    ADC = np.asarray(ADC)
    x = np.linspace(0, len(ADC[0])-1, len(ADC[0]))
    Full_ADC_Spectra.append(ADC)

print("Graphing ...")

l = int(len(names))
r = int(len(Full_ADC_Spectra[0]))
x = np.linspace(0, len(Full_ADC_Spectra[0][0])-1, len(Full_ADC_Spectra[0][0]))
s = int(np.ceil(np.sqrt(l)))

# with PdfPages(r'D:\14GeV\Pedestal_Plots.pdf') as export_pdf:
with PdfPages(r'C:\PhysicsProcessing\7.7_Ped_check\Pedestal_Plots_1.pdf') as export_pdf:
    # for i in range(r):
    for i in [385]:
        print("Working on tile ", EPDTile(i))
        fig, ax = plt.subplots(s, s, figsize=(10, 10), constrained_layout=True)
        fig.suptitle(EPDTile(i), fontsize=20)
        for j in range(l):
            if (j+1) % 10 == 0:
                print("Run", j+1, "of", l)
            u = j % s
            v = int(j/s)
            ax[v, u].plot(x, Full_ADC_Spectra[j][i])
            ax[v, u].set_title(names[j], fontsize=10)
            ax[v, u].set_xlim([0, 200])
        # fig.tight_layout()
        export_pdf.savefig()
        plt.close()

print("Boom: roasted!")
