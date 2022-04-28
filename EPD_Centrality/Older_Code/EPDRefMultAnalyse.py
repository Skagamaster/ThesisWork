import numpy as np
from numpy import loadtxt
from keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt
import os

# This is the Keras model from EPDRefMultML.py.
model = load_model(r'D:\27gev_production\data\model.h5')

# Here is the data, separated into east and west rings (1-16), Vz, and TPC refMult.
data = np.loadtxt(r'D:\27gev_production\array.txt')
refMult = data[:, 0]
maxRef = max(refMult)
# This is from the split used to train the model.
splitter = int(len(refMult)/4)
refMultTrain = refMult[:3*splitter]
refMultTest = refMult[-splitter:]

# The model uses a sigmoid, so we need to rescale the predictions.
predictions = 500*model.predict(data[:, 1:])
predictionsTrain = 500*model.predict(data[:5*splitter, 1:])
predictionsTest = 500*model.predict(data[-splitter:, 1:])

# Use this to print out a list of some predictions vs TPC refMult.
'''
for i in range(15):
    print('Result for trial %d => %.1f (expected %.1f)' %
          (i, predictions[i], data[i, 0]))
'''

# Now to make lists of refMult from the TPC and from the predictions for the model using EPD data and Vz.
# I will separate these out by centrality using the following definitions:
EPDref = []
TPCref = []
cent = []

for i in range(10):
    Enew = []
    Tnew = []
    Cnew = []
    EPDref.append(Enew)
    TPCref.append(Tnew)
    cent.append(Cnew)

# These are our centrality ranges.
cent[0] = np.arange(0, 20)      # >80%
cent[1] = np.arange(20, 32)     # 70-80%
cent[2] = np.arange(32, 47)     # 60-70%
cent[3] = np.arange(47, 68)     # 50-60%
cent[4] = np.arange(68, 94)     # 40-50%
cent[5] = np.arange(94, 129)    # 30-40%
cent[6] = np.arange(129, 173)   # 20-30%
cent[7] = np.arange(173, 231)   # 10-20%
cent[8] = np.arange(231, 268)   # 5-10%
cent[9] = np.arange(268, maxRef)   # 0-5%
cent.append([">80%", "70-80%", "60-70%", "50-60%", "40-50%",
             "30-40%", "20-30%", "10-20%", "5-10%", "0-5%"])

# for i in range(len(data[:, 0])):
#    EPDref.append(predictions[i, 0])
#    TPCref.append(data[i, 0])

for i in range(len(data[:, 0])):
    # I need to put in the cent ranges here somehow.
    for j in range(10):
        if min(cent[j]) <= refMult[i] <= max(cent[j]):
            EPDref[j].append(predictions[i])
            TPCref[j].append(refMult[i])

'''
# Just a scatter plot of the two refMults.
plt.scatter(predictions[:, 0], data[:, 0])
plt.show()
'''
# 2D histogram of the two refMults in all centrality bins.
fig, axes = plt.subplots(nrows=3, ncols=4)
spread = 20
# fig.set_title("Centrality: TPC (y) vs EPD (x)")

for i in range(3):
    for j in range(4):
        x = 4*i+j
        if i == 2 and j > 1:
            continue
        else:
            axes[i, j].hist2d(EPDref[x], TPCref[x],
                              bins=[int(10/(x+1))*len(cent[x]), len(cent[x])],
                              range=[[min(cent[x])-spread, max(cent[x])+spread],
                                     [min(cent[x]), max(cent[x])]],
                              cmin=0.01,
                              cmap=plt.cm.get_cmap("jet"))
            axes[i, j].set_title(cent[10][x])
plt.show()

# Saving the refmults for easier analysis elsewhere.
f = open(r'D:\27gev_production\data\DisplayFiles\EPDforBNL\epd.txt', 'w')
for x in range(len(predictions[:, 0])):
    f.write(str(predictions[x, 0])+"\n")
f.close()
g = open(r'D:\27gev_production\data\DisplayFiles\EPDforBNL\tpc.txt', 'w')
for x in range(len(data[:, 0])):
    g.write(str(data[x, 0])+"\n")
g.close()
