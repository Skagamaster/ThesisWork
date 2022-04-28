# \author Skipper Kagamaster
# \date 06/24/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
"""
The purpose of this code is to make fits from EPD ring sums to Refmult3 in both
data and UrQMD simulation.
"""

import uproot as up
import awkward as ak
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils.generic_utils import get_custom_objects
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras.callbacks import EarlyStopping
from tensorflow import keras as tfk
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import skew, kurtosis
import os

# Setup the model for the neural network. "Linear" is just
# a normal, linear fit (not really ML).
# I have a swish function in the definitions, but you can
# easily change it to something more standard, like RELU.
# I'd stay away from sigmoid or tanh for this fit, though.
# Choose your activation function.
# actFunc = 'linear'
# actFunc = "relu"
actFunc = "swish"
# actFunc = "mish"
# actFunc = "bose"
epochs = 100
batch_size = 300


# Let's make a Swish function, since it's not in Keras yet.
class Swish(Activation):

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'


def swish(x, beta):
    return x * K.sigmoid(beta * x)


# Import and sort the distributions.
proton_params = np.loadtxt(r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results\BigList\out_1.txt")
UrQMD_params = up.open(r"D:\UrQMD_cent_sim\15\CentralityNtuple.root")
ref_data = proton_params[:, 1]
print(ref_data)
rings_data = []
for i in range(3, 35):
    rings_data.append(proton_params[:, i])
rings_data = np.array(rings_data)
data = np.vstack((ref_data, rings_data))

ref_sim = ak.to_numpy(UrQMD_params['Rings']['RefMult3'].array())
rings_sim = []
for i in range(1, 17):
    rings_sim.append(ak.to_numpy(UrQMD_params['Rings']['r{0:0=2d}'.format(i)].array()))
rings_sim = np.array(rings_sim)
sim = np.vstack((ref_sim, rings_sim))

# Use the single perceptron if you just want to do a linear
# weight. Use the others if you would like to do something
# more complicated with a non-linear correlation.
model = Sequential()
if actFunc == 'linear':
    model.add(Dense(1, input_dim=32))
else:
    model.add(Dense(32, input_shape=(32,), activation='{0}'.format(actFunc)))
    model.add(Dense(128, activation='{0}'.format(actFunc)))
    model.add(Dense(1, activation='{0}'.format(actFunc)))
model.compile(loss='mse', optimizer='Nadam', metrics=["mse"])
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, restore_best_weights=True)
model.fit(rings_data.T, data[0].T, validation_split=0.2,  callbacks=[early_stopping],
          epochs=epochs, batch_size=batch_size)
_, accuracy = model.evaluate(rings_data.T, ref_data)
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

os.chdir(r'C:\Users\dansk\Documents\Thesis\2021_DNP_Work\ML_fits')

model.save("{0}Fitmodel_single.h5".format(actFunc))

for i in range(len(weights)):
    np.savetxt("{0}Weights{1}_single.txt".format(actFunc, i), weights[i], fmt='%f')
for i in range(len(biases)):
    np.savetxt("{0}Biases{1}_single.txt".format(actFunc, i), biases[i], fmt='%f')

plot_model(model, to_file="{0}Model_single.png".format(actFunc), show_shapes=True)

predictions = model.predict(data[1:].T)
for i in range(15):
    print('Result for trial %d => %.1f (expected %.1f)' %
          (i, predictions[i], data[0][i]))
predictions = predictions.flatten()
predictions = np.array(predictions)
np.save('{}predictions_single.npy'.format(actFunc), predictions)
plt.hist2d(predictions, ref_data, bins=[100, 100],
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

plt.title(r"RefMult3 vs $X_{\zeta',W}$")
plt.xlabel(r"$X_{\zeta',W}$")
plt.ylabel("RefMult3")
plt.colorbar()

plt.show()
