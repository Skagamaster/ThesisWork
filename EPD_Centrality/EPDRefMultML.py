# This is a machine learning tool to fit EPD centrality.
# It can be configured to fit for UrQMD, data, or what have
# you. Import the data set, separate it into training and
# testing data, set up the fitting MLP machine, and go!

# -Skipper Kagamaster

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import matplotlib as mpl
from scipy.stats import norm
import uproot as up
import os
from matplotlib.colors import LogNorm
from numba import vectorize
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils.generic_utils import get_custom_objects
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras.callbacks import EarlyStopping
from tensorflow import keras as tfk
import definitions as dfn
from definitions import Swish, swish, Mish, mish, Bose, bose, ML_run


# Let's make a Swish function, since it's not in Keras yet.
class Swish(Activation):

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'


def swish(x, beta=2):
    return x * K.sigmoid(beta * x)


fs = (32, 18)  # lets make all our figures 16 by 9

# Numpy array for all the ring values (1-16 East side, 17-32 West side, vzvpd)
# rings = np.load(r'C:\Users\dansk\Documents\Thesis\Protons\WIP\QA_4_14_2021\protons\total\ML_sets\rings.npy',
#                allow_pickle=True)
# refmult3 = np.load(r'C:\Users\dansk\Documents\Thesis\Protons\WIP\QA_4_14_2021\protons\total\ML_sets\refmult3.npy',
#                   allow_pickle=True)
# rings = rings.astype('float32')
# refmult3 = refmult3.astype('float32')
# input_dim = 33

# Loading the pickled data from the pandas array using Yu's cuts:
data = pd.read_pickle(r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results\BigList\out_all.pkl")
refmult3 = data['refmult'].to_numpy()
ring_set = np.linspace(1, 32, 32).astype("int")
rings = []
for i in ring_set:
    rings.append(data["ring{}".format(i)].to_numpy())
rings = np.array(rings).T
r = len(rings)
r = int(r/4)
rings = rings[:r]
refmult3 = refmult3[:r]
input_dim = 32

# Setup the model for the neural network and execute!
# I have a swish function in the definitions, but you can
# easily change it to something more standard, like RELU.
# I'd stay away from sigmoid or tanh for this fit, though.

# Choose your activation function.
# actFunc = 'linear'
# actFunc = "relu"
# actFunc = "swish"
# actFunc = "mish"
# actFunc = "bose"

# Use the single perceptron if you just want to do a linear
# weight. Use the others if you would like to do something
# more complicated with a non-linear correlation.
"""
model = Sequential()
if actFunc == 'linear':
    model.add(Dense(1, input_dim=input_dim))
else:
    model.add(Dense(32, input_shape=(input_dim,), activation='{0}'.format(actFunc)))
    model.add(Dense(128, activation='{0}'.format(actFunc)))
    model.add(Dense(1, activation='{0}'.format(actFunc)))
model.compile(loss='logcosh',
              optimizer='Nadam', metrics=["mae"])
early_stopping = EarlyStopping(
    monitor='val_loss', min_delta=0.01, patience=8,
    restore_best_weights=True)
model.fit(rings, refmult3, validation_split=0.2,  callbacks=[early_stopping],
          epochs=100, batch_size=3000)
_, accuracy = model.evaluate(rings, refmult3)
if accuracy < 0.03:
    print("Accuracy: %f" % (accuracy*100), r'%. You win!')
else:
    print("Accuracy: %f" % (accuracy * 100), r'%. Your life is a lie.')

weights = []
biases = []
for i in range(len(model.layers)):
    weights.append(model.layers[i].get_weights()[0])
    biases.append(model.layers[i].get_weights()[1])
weights = np.asarray(weights)
biases = np.asarray(biases)
"""
os.chdir(r'C:\Users\dansk\Documents\Thesis\2021_DNP_Work\ML_fits\9_2021')
"""
model.save("{0}Fitmodel.h5".format(actFunc))

for i in range(len(weights)):
    np.savetxt("{0}Weights{1}.txt".format(actFunc, i),
               weights[i], delimiter=',', newline='}, \n {',
               fmt='%f')
for i in range(len(biases)):
    np.savetxt("{0}Biases{1}.txt".format(actFunc, i),
               biases[i], delimiter=',', newline=",",
               fmt='%f')

plot_model(model, to_file="{0}Model.png".format(actFunc), show_shapes=True)

predictions = model.predict(rings)
"""
actFunc = ['linear', 'relu', 'swish', 'mish']
predictions = []
for i in actFunc:
    predictions.append(ML_run(rings, refmult3, actFunc=i))
predictions = np.array(predictions)
for i in range(4):
    print("ActFunc =", actFunc[i])
    for j in range(15):
        print('Result for trial %d => %.1f (expected %.1f)' %
              (j, predictions[i][j], refmult3[j]))

fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
for i in range(2):
    for j in range(2):
        x = 2*i + j
        count, binsX, binsY = np.histogram2d(refmult3, predictions[x],
                                             bins=700, range=((0, 700), (0, 700)))
        X, Y = np.meshgrid(binsX[:-1], binsY[:-1])
        im = ax[i, j].pcolormesh(X, Y, count, norm=LogNorm(), cmap="jet", shading="auto")
        ax[i, j].set_title(actFunc[x])
        ax[i, j].set_xlabel("RefMult3")
        ax[i, j].set_ylabel(r"$X_{\zeta',W}$")
        fig.colorbar(im, ax=ax[i, j])
plt.show()

"""
predictions = predictions.flatten()
predictions = np.array(predictions)
np.save('{}predictions.npy'.format(actFunc), predictions)

plt.hist2d(refmult3, predictions, bins=(700, 700), range=((0, 700), (0, 700)), cmin=1, cmap='jet', norm=LogNorm())
plt.title(r"RefMult3 vs $X_{\zeta',W}$")
plt.ylabel(r"$X_{\zeta',W}$")
plt.xlabel("RefMult3")
plt.colorbar()

plt.show()
"""
