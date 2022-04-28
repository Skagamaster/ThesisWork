import numpy as np
import uproot as up
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

# Directory for saving.
os.chdir(r"C:\Users\dansk\Documents\Thesis\Tristan\Unsupervised_Learning\Supervised_Data")


# Let's make a Swish function, since it's not in Keras yet.
class Swish(Activation):

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'


def swish(x, beta=2):
    return x * K.sigmoid(beta * x)


# Here is the 7.7 GeV simulation data.
file = up.open(r"C:\Users\dansk\Downloads\simulated_data.root")
data_len = int(len(file['ring_sums'].member("fElements")) / 16)
rings = np.reshape(file["ring_sums"].member("fElements"), (16, data_len)).T
tpc = file["tpc_multiplicity"].member("fElements")
b = file["impact_parameter"].member("fElements")
target = b

print(len(b), len(rings))

# Setup the model for the neural network and execute!
model = Sequential()
# model.add(Dense(1, input_dim=16))
rings = np.expand_dims(rings, axis=2)
model.add(Conv1D(128, kernel_size=2, activation='swish', input_shape=(16, 1)))
model.add(Conv1D(256, 5, activation='swish'))
model.add(Flatten())
# model.add(Dense(32, input_shape=(16,), activation='swish'))
model.add(Dense(128, activation='swish'))
# model.add(Dense(256, activation='swish'))
model.add(Dense(1, activation='swish'))
model.compile(loss='mae',
              optimizer='Nadam', metrics=['mse'])
early_stopping = EarlyStopping(
    monitor='val_loss', min_delta=0.0005, patience=15,
    restore_best_weights=True)
model.fit(rings, target, validation_split=0.1, epochs=300, batch_size=500,
          callbacks=[early_stopping])
_, accuracy = model.evaluate(rings, target)
if accuracy < 0.15:
    print("Accuracy: %f" % (accuracy*100), r'%. Your life is a lie.')
else:
    print("Accuracy: %f" % (accuracy*100), r'%. You win!')

predictions = model.predict(rings)
for i in range(15):
    print('Result for trial %d => %.1f (expected %.1f)' %
          (i, predictions[i], target[i]))
predictions = predictions.flatten()
np.save("MLP_predictions.npy", predictions)

plt.hist2d(target, predictions, bins=100, cmin=1)
plt.colorbar()
plt.show()

weights = []
biases = []
for i in range(len(model.layers)):
    weights.append(model.layers[i].get_weights()[0])
    biases.append(model.layers[i].get_weights()[1])
weights = np.asarray(weights, dtype=object)
biases = np.asarray(biases, dtype=object)

model.save("swish_sim_model.h5")

for i in range(len(weights)):
    np.savetxt("swish_sim_weights{}.txt".format(i), weights[i], delimiter=',', newline='}, \n {')
for i in range(len(biases)):
    np.savetxt("swish_sim_biases{}.txt".format(i), biases[i], delimiter=',', newline=",")
