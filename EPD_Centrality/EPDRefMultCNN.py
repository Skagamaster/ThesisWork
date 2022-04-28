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
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, Conv1D
from keras.utils.generic_utils import get_custom_objects
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras.callbacks import EarlyStopping
import definitions as dfn
from definitions import Swish, swish, Mish, mish, Bose, bose
# Let's make a Swish function, since it's not in Keras yet.


class Swish(Activation):

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'


def swish(x, beta=2):
    return x * K.sigmoid(beta * x)


fs = (32, 18)  # lets make all our figures 16 by 9

# Numpy array for all the ring values (1-16 East side, 17-32 West side, vzvpd)
rings = np.load(r'C:\Users\dansk\Documents\Thesis\Protons\WIP\QA_4_14_2021\protons\total\ML_sets\rings.npy',
                allow_pickle=True)
refmult3 = np.load(r'C:\Users\dansk\Documents\Thesis\Protons\WIP\QA_4_14_2021\protons\total\ML_sets\refmult3.npy',
                   allow_pickle=True)
rings = rings.astype('float32')
refmult3 = refmult3.astype('float32')
r = len(rings) // 5
rings = rings[:r]
refmult3 = refmult3[:r]
rings = np.expand_dims(rings, axis=2)

# Setup the model for the neural network and execute!
# I have a swish function in the definitions, but you can
# easily change it to something more standard, like RELU.
# I'd stay away from sigmoid or tanh for this fit, though.

# Choose your activation function.
actFunc = "relu"
#actFunc = "swish"
#actFunc = "mish"
#actFunc = "bose"

# Use the single perceptron if you just want to do a linear
# weight. Use the others if you would like to do something
# more complicated with a non-linear correlation.
model = Sequential()
model.add(Conv1D(128, kernel_size=3, activation='{0}'.format(
    actFunc), input_shape=(33, 1)))
model.add(Conv1D(256, 5, activation='{0}'.format(actFunc)))
model.add(Flatten())
#model.add(Dense(16, input_shape=(33,), activation='{0}'.format(actFunc)))
#model.add(Dense(1024, activation='{0}'.format(actFunc)))
model.add(Dense(512, activation='{0}'.format(actFunc)))
model.add(Dense(1, activation='{0}'.format(actFunc)))
model.compile(loss='mse',
              optimizer='Nadam', metrics=['accuracy'])
early_stopping = EarlyStopping(
    monitor='val_accuracy', min_delta=0, patience=10,
    restore_best_weights=True)
model.fit(rings, refmult3, validation_split=0.2,  callbacks=[early_stopping],
          epochs=100, batch_size=80000)
_, accuracy = model.evaluate(rings, refmult3)
if accuracy < 0.03:
    print("Accuracy: %f" % (accuracy * 100), r'%. You win!')
else:
    print("Accuracy: %f" % (accuracy*100), r'%. Your life is a lie.')

"""
weights = []
biases = []
for i in range(len(model.layers)):
    weights.append(model.layers[i].get_weights()[0])
    biases.append(model.layers[i].get_weights()[1])
weights = np.asarray(weights)
biases = np.asarray(biases)
"""

os.chdir(r'C:\Users\dansk\Documents\Thesis\Protons\WIP\QA_4_14_2021\protons\total\ML_sets')

model.save("CNN{0}model_cut.h5".format(actFunc))

"""
for i in range(len(weights)):
    np.savetxt("{0}TESTWeights{1}_cut.txt".format(actFunc, i),
               weights[i], delimiter=',', newline='}, \n {')
for i in range(len(biases)):
    np.savetxt("{0}TESTBiases{1}_cut.txt".format(actFunc, i),
               biases[i], delimiter=',', newline=",")
"""

saveLoc = r"C:\Users\dansk\Documents\Thesis\Protons\WIP\QA_4_14_2021\protons\total\ML_sets\\"
plot_model(
    model, to_file=saveLoc+"CNN{0}Model_cut.png".format(actFunc), show_shapes=True)
predictions = model.predict(rings)
for i in range(15):
    print('Result for trial %d => %.1f (expected %.1f)' %
          (i, predictions[i], refmult3[i]))
predictions = predictions.flatten()
plt.hist2d(predictions, refmult3, bins=[100, 100],
           cmin=0.1, cmap=plt.cm.get_cmap("jet"))

# joint_kws = dict(gridsize=250)

# sns.jointplot(predictions, 500*data[:, 0],
#              # kind="hex",
#              kind='kde',
#              height=8,
#              ratio=10,
#              space=0,
#              # color='r',
#              cmap='jet',
#              joint_kws=joint_kws
#              )

plt.title(r"RefMult1 vs $X_{\zeta',W}$, Test Set")
plt.xlabel(r"$X_{\zeta',W}$")
plt.ylabel("RefMult1")
plt.colorbar()

plt.show()
