import numpy as np
from numpy import loadtxt
from keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from scipy.stats import norm

os.chdir(r'D:\27gev_production\data\UrQMD')

# This is the Keras model from EPDrefMultML.py.
model = load_model(r'D:\27gev_production\data\UrQMD\relufitmodelUrQMD.h5')

# Here is the data, separated into rings (1-16) and impact parameter (B).
data = np.loadtxt(r'D:\27gev_production\arrayUrQMD.txt')
B = data[:, -1:]
splitter = int(len(B)/4)  # Split from the ML training.
maxRef = max(B)
BTrain = B[:splitter*3]
BTest = B[-splitter:]
B = B.flatten()
BTrain = BTrain.flatten()
BTest = BTest.flatten()

# Raw nMIP sums from all rings.
nMIP = np.empty(len(data))
for i in range(len(nMIP)):
    nMIP[i] = np.sum(data[i, :-1])

#nMIP = nMIP.flatten()
nMIPTrain = nMIP[:splitter*3]
nMIPTest = nMIP[-splitter:]
nMIPTrain = nMIPTrain.flatten()
nMIPTest = nMIPTest.flatten()

# Predictions from our trained model.
predictions = model.predict(data[:, :-1])
predictions = predictions.flatten()
predictionsTrain = model.predict(data[:splitter*3, :-1])
predictionsTrain = predictionsTrain.flatten()
predictionsTest = model.predict(data[-splitter:, :-1])
predictionsTest = predictionsTest.flatten()

# Display for the fit in terminal.
model1 = sm.OLS(BTest, predictionsTest).fit()
print(model1.summary())

# Delta for the predictions.
deltaTrain = BTrain-predictionsTrain
deltaTest = BTest-predictionsTest
meanTrain, stdTrain = norm.fit(deltaTrain)
meanTest, stdTest = norm.fit(deltaTest)

# 2D histograms of predictive correlation.

plt.figure(figsize=(16, 9))
plt.hist2d(B, nMIP, range=((3, 10), (100, 800)),
           bins=[200, 200], cmin=0.01, cmap=plt.cm.get_cmap("jet"))
plt.title(
    r"Raw nMIP vs B: UrQMD Au+Au $\sqrt{s}$=27 GeV", fontsize=32)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("nMIP Sum (raw)", fontsize=20)
plt.ylabel("Impact Parameter (B)", fontsize=20)
plt.colorbar()
plt.tight_layout()
plt.savefig('nMIPvBTrainUrQMD.png')
plt.close()

plt.figure(figsize=(16, 9))
plt.hist2d(predictionsTest, BTest, range=((0, 14), (3, 10)),
           bins=[200, 200], cmin=0.01, cmap=plt.cm.get_cmap("jet"))
plt.title(
    r"WnMIP vs B (Test): UrQMD Au+Au $\sqrt{s}$=27 GeV", fontsize=32)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("WnMIP", fontsize=20)
plt.ylabel("B", fontsize=20)
plt.colorbar()
plt.tight_layout()
plt.savefig('RELUTestUrQMD.png')
plt.close()

plt.figure(figsize=(16, 9))
plt.hist2d(predictionsTrain, BTrain, range=((0, 14), (3, 10)),
           bins=[200, 200], cmin=0.01, cmap=plt.cm.get_cmap("jet"))
plt.title(
    r"WnMIP vs B (Train): UrQMD Au+Au $\sqrt{s}$=27 GeV", fontsize=32)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("WnMIP", fontsize=20)
plt.ylabel("B", fontsize=20)
plt.colorbar()
plt.tight_layout()
plt.savefig('RELUTrainUrQMD.png')
plt.close()

# Scatter plot for deltas.
plt.figure(figsize=(16, 9))
plt.hist(deltaTrain, bins=200, range=(-5, 5),
         histtype='step', normed=True, linewidth=3,
         label='Trained Data')
plt.hist(deltaTest, bins=200, range=(-5, 5),
         histtype='step', normed=True, linewidth=3,
         label='Test Data', color='red')
plt.title(
    r"$\Delta$: B vs WnMIP", fontsize=32)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc='upper right')
plt.xlabel(r'$\Delta$', fontsize=20)
plt.ylabel("nCounts", fontsize=20)
# x = np.linspace(-100, 100, 50)
# y = norm.pdf(x, meanTest, stdTest)
# plt.plot(x, y)
plt.savefig('DeltaRELUUrQMD.png')
print(meanTrain-meanTest, stdTrain-stdTest)
plt.close()
'''
# joint_kws = dict(gridsize=250)
plt.figure(figsize=(12, 12))
sns.jointplot(predictionsTrain, BTrain,
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
