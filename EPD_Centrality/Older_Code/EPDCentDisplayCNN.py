import numpy as np
from numpy import loadtxt
from keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from scipy.stats import norm
import pandas as pd
from definitions import Swish, swish, Mish, mish

# Activation function to investigate
#actFunc = 'relu'
#actFunc = 'swish'
actFunc = 'mish'

os.chdir(r'D:\27gev_production\2020_Version\ML\Images')

# This is the Keras model from EPDRefMultML.py.
saveLoc = r"D:\27gev_production\2020_Version\ML\\"
model = load_model(saveLoc + "CNN{0}TESTmodel_cut.h5".format(actFunc))

# Here is the data, separated into east and west rings (1-16), VzVPD,
# and TPC refMult.
data = pd.read_pickle(r'D:\27gev_production\pandasdatacut.pkl')
rings = np.load(r'D:\27gev_production\ringscut.npy', allow_pickle=True)
# Get the raw, TnMIP values
nMIP = []
for i in rings:
    x = np.sum(i[:-1])
    nMIP.append(x)
nMIP = np.asarray(nMIP)

index = int(len(data))
split = 4
indexT = int(index/split)
RefMult = data.RefMult1.to_numpy()
VzVPD = data.VzVPD.to_numpy()
learnSet = []
for i in range(int(len(VzVPD))):
    learnSet.append(np.append(rings[i], VzVPD[i]))
learnSet = np.asarray(learnSet)
learnSet = np.expand_dims(learnSet, axis=2)

# Now we split our data into training and testing sets, with the testing
# set being 20% of our total data set.
learnSetTrain = learnSet[:-indexT]
learnSetTest = learnSet[-indexT:]
RefMultTrain = RefMult[:-indexT]
RefMultTest = RefMult[-indexT:]

# Predictions from our trained model.
predictions = model.predict(learnSet)
predictions = predictions.flatten()
predictionsTrain = model.predict(learnSetTrain)
predictionsTrain = predictionsTrain.flatten()
predictionsTest = model.predict(learnSetTest)
predictionsTest = predictionsTest.flatten()

# Display for the fit in terminal.
model1 = sm.OLS(RefMultTest, predictionsTest).fit()
print(model1.summary())

# Delta for the predictions.
deltaTrain = RefMultTrain-predictionsTrain
deltaTest = RefMultTest-predictionsTest
meanTrain, stdTrain = norm.fit(deltaTrain)
meanTest, stdTest = norm.fit(deltaTest)

# 2D histograms of predictive correlation.
plt.figure(figsize=(12, 12))
plt.hist2d(nMIP, RefMult, range=((0, 700), (0, 150)),
           bins=[150, 150], cmin=0.01, cmap=plt.cm.get_cmap("jet"))
plt.title(
    r"$X_{\zeta'}$ vs RefMult1: Au+Au $\sqrt{s}$=27 GeV", fontsize=32)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel(r"$X_{\zeta'}$", fontsize=20)
plt.ylabel("RefMult1", fontsize=20)
plt.colorbar()
plt.tight_layout()
plt.savefig('TESTTnMIPvRef1.png')
plt.close()

plt.figure(figsize=(12, 12))
plt.hist2d(predictions, RefMult, range=((0, 150), (0, 150)),
           bins=[150, 150], cmin=0.01, cmap=plt.cm.get_cmap("jet"))
plt.title(
    r"$X_{W,\zeta'}$ vs RefMult1, Au+Au $\sqrt{s}$=27 GeV", fontsize=32)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel(r"$X_{W,\zeta'}$", fontsize=20)
plt.ylabel("RefMult1", fontsize=20)
plt.colorbar()
plt.tight_layout()
plt.savefig('TESTRef1vZeta{0}.png'.format(actFunc))
plt.close()

# Scatter plot for deltas.
plt.figure(figsize=(12, 12))
plt.hist(deltaTrain, bins=400, range=(-100, 100),
         histtype='step', density=True, linewidth=3,
         label='Trained Data')
plt.hist(deltaTest, bins=400, range=(-100, 100),
         histtype='step', density=True, linewidth=3,
         label='Test Data', color='red')
plt.title(
    r"$\Delta$ for Refmult, TPC v EPD", fontsize=32)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc='upper right')
plt.xlabel(r'RefMult $\Delta$', fontsize=20)
plt.ylabel("Counts (Normalised)", fontsize=20)
# x = np.linspace(-100, 100, 50)
# y = norm.pdf(x, meanTest, stdTest)
# plt.plot(x, y)
plt.savefig('TESTDeltaref1{0}.png'.format(actFunc))
plt.close()
'''
# joint_kws = dict(gridsize=250)
plt.figure(figsize=(12, 12))
sns.jointplot(predictionsTrain, refMultTrain,
              # kind="hex",
              kind='kde',
              height=8,
              ratio=10,
              space=0,
              # color='r',
              cmin=0.01,
              cmap='jet',
              # joint_kws=joint_kws
              )
plt.show()
'''
