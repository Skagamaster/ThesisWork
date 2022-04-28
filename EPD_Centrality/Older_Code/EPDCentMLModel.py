import numpy as np
import math
from numpy import loadtxt
from keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm

# This is the Keras model from EPDRefMult.py.
model = load_model(r'D:\27gev_production\data\model.h5')

# Now to create arrays for the weights and biases from the model.
# There are 3 layers: Input->Hidden->Output. All are RELU.
weights = []
biases = []
for i in range(len(model.layers)):
    weights.append(model.layers[i].get_weights()[0])
    biases.append(model.layers[i].get_weights()[1])

weights = np.asarray(weights)
biases = np.asarray(biases)

# Here is the data, separated into east and west rings (1-16), Vz, and TPC refMult.
data = np.loadtxt(r'D:\27gev_production\array.txt')
refMult = data[:, 0]
splitter = int(len(refMult)/4)  # Split from the ML training.
maxRef = max(refMult)
refMultTrain = refMult[:splitter*4]
refMultTest = refMult[-splitter:]

# Here are the predictions from the trained model.
predictions = model.predict(data[:, 1:])
predictions = predictions.flatten()
predictionsTrain = model.predict(data[:splitter*4, 1:])
predictionsTrain = predictionsTrain.flatten()
predictionsTest = model.predict(data[-splitter:, 1:])
predictionsTest = predictionsTest.flatten()

# Here is the setup for the manual reconstruction of the model
# to test this as a class input for ROOT.
nMIPs = data[:, 1:]

# This is the input layer.
y = np.matmul(nMIPs, weights[0])+biases[0][None, :]
y = np.where(y < 0, 0, y)
# Hidden layer.
z = np.matmul(y, weights[1])+biases[1][None, :]
z = np.where(z < 0, 0, z)
# Output layer.
predictions1 = np.matmul(z, weights[2])+biases[2][None, :]


# Now to test the predictions with the manual model against the ML one.
# If done correctly, they should match perfectly.
for i in range(330, 340):
    print('Result for trial %d => %.1f (expected %.1f)' %
          (i, predictions[i], refMult[i]))
    print('Extrapolation for trial %d => %.1f' %
          (i, predictions1[i]))
