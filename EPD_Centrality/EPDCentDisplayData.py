import numpy as np
from numpy import loadtxt
from keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from scipy.stats import norm
import pandas as pd
from definitions import Swish, swish

# Enter the model you want to analyse.
enterModel = r"\CNNrelumodel_cut.h5"
modelName = "CNN"
# Directory for saving images.
imageSave = r'D:\27gev_production\2020_Version\ML\Images'
# Directory for loading data.
dataLoad = r'D:\27gev_production\2020_Version\ML'

# This is the Keras model from EPDRefMultML.py.
loadModel = f"{enterModel}.h5"
model = load_model(dataLoad + enterModel)

# Here is the 27 GeV simulation data.
data = pd.read_pickle(r'D:\27gev_production\pandasdatacut.pkl')
# Numpy array for all the ring values (1-16 East side, 17-32 West side)
rings = np.load(r'D:\27gev_production\ringscut.npy', allow_pickle=True)

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
nMIP = []
for i in rings:
    nMIP.append(np.sum(i))

# Now we split our data into training and testing sets, with the testing
# set being 20% of our total data set (as done in the ML fit).
learnSetTrain = rings[:-indexT]
learnSetTest = rings[-indexT:]
RefMultTrain = RefMult[:-indexT]
RefMultTest = RefMult[-indexT:]

# Predictions from our trained model.
predictions = model.predict(learnSet).flatten()
predictionsTrain = predictions[:-indexT]
predictionsTest = predictions[-indexT:]


# Display for the fit in terminal.
model1 = sm.OLS(RefMultTest, predictionsTest).fit()
print(model1.summary())

# Delta for the predictions.
deltaTrain = RefMultTrain-predictionsTrain
deltaTest = RefMultTest-predictionsTest
meanTrain, stdTrain = norm.fit(deltaTrain)
meanTest, stdTest = norm.fit(deltaTest)

# 2D histograms of predictive correlation.
os.chdir(imageSave)

plt.figure(figsize=(12, 12))
plt.hist2d(nMIP, RefMult,  range=((0, 150), (0, 150)),
           bins=[150, 150], cmin=0.01, cmap=plt.cm.get_cmap("jet"))
plt.title(
    r"$X_{\zeta'}$ vs RefMult1: UrQMD $\sqrt{s}$=27 GeV", fontsize=32)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel(r"$X_{\zeta'}$", fontsize=20)
plt.ylabel("RefMult1", fontsize=20)
plt.colorbar()
plt.tight_layout()
plt.savefig(f'TnMIPvRef1{modelName}.png')
plt.close()

plt.figure(figsize=(12, 12))
plt.hist2d(predictions, RefMult,  range=((0, 150), (0, 150)),
           bins=[150, 150], cmin=0.01, cmap=plt.cm.get_cmap("jet"))
plt.title(
    r"$X_{W,\zeta'}$ vs RefMult1, Au+Au $\sqrt{s}$=27 GeV", fontsize=32)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel(r"$X_{W,\zeta'}$", fontsize=20)
plt.ylabel("RefMult1", fontsize=20)
plt.colorbar()
plt.tight_layout()
plt.savefig(f'Ref1vZeta{modelName}.png')
plt.close()

# Scatter plot for deltas.
plt.figure(figsize=(12, 12))
plt.hist(deltaTrain, bins=400,  range=(-100, 100),
         histtype='step', density=True, linewidth=3,
         label='Trained Data')
plt.hist(deltaTest, bins=400,  range=(-100, 100),
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
plt.savefig(f'Delta{modelName}.png')
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
