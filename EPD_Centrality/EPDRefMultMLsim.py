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
import definitions as dfn
from definitions import Swish, swish

fs = (32, 18)  # lets make all our figures 16 by 9

# Here is the 27 GeV simulation data.
data = pd.read_pickle(r'D:\UrQMD_cent_sim\27\pandassim.pkl')

index = int(len(data))
split = 4
indexT = int(index/split)
RefMult = data.refMult.to_numpy()
ringNames = data.columns[2:]
rings = []
for i in range(16):
    rings.append(data.loc[:, ringNames[i]].to_numpy())
rings = np.asarray(rings).T

# Now we split our data into training and testing sets, with the testing
# set being 20% of our total data set.
# (I figured out how to do this automatically, but this is still
# usefull for graphing, so I left it in.)
learnSetTrain = rings[:-indexT]
learnSetTest = rings[-indexT:]
RefMultTrain = RefMult[:-indexT]
RefMultTest = RefMult[-indexT:]

# Setup the model for the neural network and execute!
model = Sequential()
model.add(Dense(1, input_dim=16))
#model.add(Dense(256, input_shape=(16,), activation='swish'))
#model.add(Dense(64, activation='swish'))
#model.add(Dense(1, activation='swish'))
model.compile(loss='mean_squared_error',
              optimizer='Nadam', metrics=['accuracy'])
early_stopping = EarlyStopping(
    monitor='val_accuracy', min_delta=0.0001, patience=5,
    restore_best_weights=True)
model.fit(rings, RefMult, validation_split=0.2, callbacks=[early_stopping],
          epochs=100, batch_size=300)
_, accuracy = model.evaluate(learnSetTest, RefMultTest)
if accuracy < 0.03:
    print("Accuracy: %f" % (accuracy*100), r'%. Your life is a lie.')
else:
    print("Accuracy: %f" % (accuracy*100), r'%. You win!')

predictions = model.predict(rings)
for i in range(15):
    print('Result for trial %d => %.1f (expected %.1f)' %
          (i, predictions[i], RefMult[i]))
predictions = predictions.flatten()

weights = []
biases = []
for i in range(len(model.layers)):
    weights.append(model.layers[i].get_weights()[0])
    biases.append(model.layers[i].get_weights()[1])
weights = np.asarray(weights)
biases = np.asarray(biases)

os.chdir(r'D:\27gev_production\2020_Version\ML')

model.save("LinSIMfitmodel_nocut.h5")

for i in range(len(weights)):
    np.savetxt("LinSIMWeights%s_nocut.txt" %
               i, weights[i], delimiter=',', newline='}, \n {')
for i in range(len(biases)):
    np.savetxt("LinSIMBiases%s_nocut.txt" %
               i, biases[i], delimiter=',', newline=",")

plot_model(
    model, to_file=r'D:\27gev_production\2020_Version\ML\LinSIMmodel.png', show_shapes=True)

plt.hist2d(predictions, RefMult, bins=[100, 100],
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

plt.title(r"RefMult vs $X_{\zeta',W}$, Test Set")
plt.xlabel(r"$X_{\zeta',W}$")
plt.ylabel("RefMult")
plt.colorbar()

plt.show()
