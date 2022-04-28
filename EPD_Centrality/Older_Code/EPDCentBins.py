import numpy as np
from numpy import loadtxt
from keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt
import os

# This is the Keras model from EPDRefMult.py.
model = load_model(r'D:\27gev_production\data\relufitmodelCutoff.h5')

# Here is the data, separated into east and west rings (1-16), Vz, and TPC refMult.
data = np.loadtxt(r'D:\27gev_production\arrayCutoff.txt')
refMult = data[:, 0]
split = 4  # Split from the ML training.
splitter = int(len(refMult)/split)
maxRef = max(refMult)
refMultTrain = refMult[:splitter*(split-1)]
refMultTest = refMult[-splitter:]

# Predictions from the ML model.
predictions = model.predict(data[:, 1:])
predictionsTrain = predictions[:splitter*(split-1)]
predictionsTest = predictions[-splitter:]

'''
# Use this to print out a list of some predictions vs TPC refMult.
for i in range(15):
    print('Result for trial %d => %.1f (expected %.1f)' %
          (i, predictions[i], data[i, 0]))
'''

# Now to make lists of refMult from the TPC and from the predictions for the model using EPD data and Vz.
# I will separate these out by centrality using the following definitions:
EPDrefTrain = []
TPCrefTrain = []
cent = []
EPDrefTest = []
TPCrefTest = []
Deltas = []

for i in range(10):
    EPDrefTrain.append([])
    TPCrefTrain.append([])
    EPDrefTest.append([])
    TPCrefTest.append([])
    cent.append([])
    Deltas.append([])

# These are our centrality ranges.
cent[0] = np.arange(0, 5.99)      # >80%
cent[1] = np.arange(6, 12.99)     # 70-80%
cent[2] = np.arange(13, 24.99)     # 60-70%
cent[3] = np.arange(25, 43.99)     # 50-60%
cent[4] = np.arange(44, 71.99)     # 40-50%
cent[5] = np.arange(72, 112.99)    # 30-40%
cent[6] = np.arange(113, 167.99)   # 20-30%
cent[7] = np.arange(168, 245.99)   # 10-20%
cent[8] = np.arange(246, 294.99)   # 5-10%
cent[9] = np.arange(295, maxRef)   # 0-5%
# Strings for centrality bins.
cent.append([">80%", "70-80%", "60-70%", "50-60%", "40-50%",
             "30-40%", "20-30%", "10-20%", "5-10%", "0-5%"])
# Centrality ranges for sorting.
cent.append([np.arange(80, 100), np.arange(70, 80), np.arange(60, 70),
             np.arange(50, 60), np.arange(40, 50),
             np.arange(30, 40), np.arange(20, 30),
             np.arange(10, 20), np.arange(5, 10),
             np.arange(0, 5)])

centRange = [80, 70, 60, 50, 40, 30, 20, 10, 5, 0]

# Creating EPDref and TPCref.
for i in range(len(refMultTrain)):
    for j in range(10):
        if min(cent[j]) <= refMultTrain[i] <= max(cent[j]):
            EPDrefTrain[j].append(predictionsTrain[i])
            TPCrefTrain[j].append(refMultTrain[i])

for i in range(len(refMultTest)):
    for j in range(10):
        if min(cent[j]) <= refMultTest[i] <= max(cent[j]):
            EPDrefTest[j].append(predictionsTest[i])
            TPCrefTest[j].append(refMultTest[i])
            Deltas[j].append(predictionsTest[i][0]-refMultTest[i])

meanStd = np.empty([10, 2])

for i in range(10):
    meanStd[i, 0] = np.mean(Deltas[i])
    meanStd[i, 1] = np.std(Deltas[i])

print(meanStd)

os.chdir(r'D:\27gev_production\data')
'''
# Here's a plot of the centrality bins vs EPD refMult.
plt.figure(figsize=(16, 9))
plt.hist([EPDrefTrain[0], EPDrefTrain[1], EPDrefTrain[2], EPDrefTrain[3], EPDrefTrain[4],
          EPDrefTrain[5], EPDrefTrain[6], EPDrefTrain[7], EPDrefTrain[8], EPDrefTrain[9]],
         400, histtype='barstacked', label=cent[10])
plt.yscale('log')
plt.xlim(0, 400)
plt.legend(fontsize=20, title='TPC Centrality Range', title_fontsize=20)
plt.title(
    r"Centrality: TPC vs EPD (Train), Au+Au $\sqrt{s}$=27 GeV", fontsize=28)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("EPD refMult", fontsize=20)
plt.ylabel("Counts", fontsize=20)
plt.tight_layout()
plt.savefig('refMultRELUTrain.png')
plt.close()

plt.figure(figsize=(16, 9))
plt.hist([EPDrefTrain[2], EPDrefTrain[7]], 200,
         histtype='barstacked', label=("60-70%", "10-20%"))
plt.xlim(0, 400)
plt.legend(fontsize=20, title='TPC Centrality Range', title_fontsize=20)
plt.title(
    r"Centrality: TPC vs EPD (Train), Samples", fontsize=28)
plt.axvline(x=min(cent[2]), linewidth=3, color='r')
plt.axvline(x=max(cent[2]), linewidth=3, color='r')
plt.axvline(x=min(cent[7]), linewidth=3, color='r')
plt.axvline(x=max(cent[7]), linewidth=3, color='r')
plt.axvline(x=min(cent[1]), linewidth=3, color='y')
plt.axvline(x=max(cent[3]), linewidth=3, color='y')
plt.axvline(x=min(cent[6]), linewidth=3, color='y')
plt.axvline(x=max(cent[8]), linewidth=3, color='y')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("EPD refMult", fontsize=20)
plt.ylabel("Counts", fontsize=20)
plt.tight_layout()
plt.savefig('refMultRELUTrainBO.png')
plt.close()

plt.figure(figsize=(16, 9))
plt.hist([EPDrefTest[0], EPDrefTest[1], EPDrefTest[2], EPDrefTest[3], EPDrefTest[4],
          EPDrefTest[5], EPDrefTest[6], EPDrefTest[7], EPDrefTest[8], EPDrefTest[9]],
         400, histtype='barstacked', label=cent[10])
plt.yscale('log')
plt.xlim(0, 400)
#plt.ylim(100, 2000)
plt.legend(fontsize=20, title='TPC Centrality Range', title_fontsize=20)
plt.title(
    r"Centrality: TPC vs EPD (Test), Au+Au $\sqrt{s}$=27 GeV", fontsize=28)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("EPD refMult", fontsize=20)
plt.ylabel("Counts", fontsize=20)
plt.tight_layout()
plt.savefig('refMultRELUTest.png')
plt.close()

plt.figure(figsize=(16, 9))
plt.hist([EPDrefTest[2], EPDrefTest[7]], 200,
         histtype='barstacked', label=("60-70%", "10-20%"))
plt.xlim(0, 400)
plt.legend(fontsize=20, title='TPC Centrality Range', title_fontsize=20)
plt.title(
    r"Centrality: TPC vs EPD (Test), Samples", fontsize=28)
plt.axvline(x=min(cent[2]), linewidth=3, color='r')
plt.axvline(x=max(cent[2]), linewidth=3, color='r')
plt.axvline(x=min(cent[7]), linewidth=3, color='r')
plt.axvline(x=max(cent[7]), linewidth=3, color='r')
plt.axvline(x=min(cent[1]), linewidth=3, color='y')
plt.axvline(x=max(cent[3]), linewidth=3, color='y')
plt.axvline(x=min(cent[6]), linewidth=3, color='y')
plt.axvline(x=max(cent[8]), linewidth=3, color='y')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("EPD refMult", fontsize=20)
plt.ylabel("Counts", fontsize=20)
plt.tight_layout()
plt.savefig('refMultRELUTestBO.png')
plt.close()
'''
# Histograms of the refMult delta in all centrality bins.
fig, axes = plt.subplots(nrows=3, ncols=4)
spread = 20
# fig.set_title("Centrality: TPC (y) vs EPD (x)")

for i in range(3):
    for j in range(4):
        x = 4*i+j
        if i == 2 and j > 1:
            continue
        else:
            axes[i, j].hist(Deltas[x], bins=50, histtype='step',
                            normed='True', linewidth=2)
            axes[i, j].set_title(cent[10][x])
plt.show()
plt.close()

plt.figure(figsize=(16, 9))
plt.errorbar(centRange[:], meanStd[:, 0], yerr=meanStd[:, 1], capsize=10, marker="o", markersize=12,
             mfc='red', elinewidth=4, markeredgewidth=4, linestyle="None")
plt.title(r'$\mu$ and $\sigma$ for EPD-TPC $\Delta$', fontsize=32)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Centrality Bin (start)", fontsize=20)
plt.ylabel(r"$\mu$", fontsize=20)
plt.tight_layout()
plt.show()
