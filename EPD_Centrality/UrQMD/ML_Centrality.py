import numpy as np
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

fs = (32, 18)  # lets make all our figures 16 by 9
# Import the arrays from the paper.
os.chdir(r'D:\UrQMD_cent_sim\PaperFiles\En_200\e_by_e')
files = up.open('ppkset_rap_200.root')  # For my data.
urqmd = files['AMPT_tree']

refMult = []
for i in range(4):
    refMult.append([])
#refMult[0] = urqmd.array('Imp')
refMult[0] = urqmd.array('RefMult')+urqmd.array('RefMult2')
refMult[1] = urqmd.array('RefMultEPD1')
refMult[2] = urqmd.array('RefMultEPD2')
refMult[3] = urqmd.array('RefMultEPD3')

feeder = np.transpose(np.asarray([refMult[1], refMult[2], refMult[3]]))
target = np.asarray(refMult[0])

# Other data set
# data = np.loadtxt(r'D:\UrQMD_cent_sim\19\arrayUrQMD.txt')

# Now we split our data into training and testing sets, with the testing
# set being 20% of our total data set.
index = int(len(target))
print(index)
split = 4
indexT = int(index/split)
feederTrain = feeder[:indexT*(split-1)]
feederTest = feeder[-indexT:]
targetTrain = target[:indexT*(split-1)]
targetTest = target[-indexT:]

# Setup the model for the neural network and execute!
model = Sequential()
model.add(Dense(1, use_bias=True, input_dim=3))
#model.add(Dense(32, input_dim=3, activation='relu'))
#model.add(Dense(32, activation='relu'))
#model.add(Dense(1, activation='relu'))
model.compile(loss='mae',
              optimizer='Nadam', metrics=['mse'])
model.fit(feederTrain, targetTrain,
          epochs=200, batch_size=40000)

_,  accuracy = model.evaluate(feederTest, targetTest)
if accuracy > 0.89:
    print("Mean squared error: %f" % (accuracy), r'Your life is a lie.')
else:
    print("Mean squared error: %f" % (accuracy), r'You win!')

predictions = model.predict(feeder)
for i in range(200000, 200015):
    print('Result for trial %d => %.1f (expected %.1f)' %
          (i, predictions[i], target[i]))
predictions = predictions.flatten()

weights = []

biases = []
for i in range(len(model.layers)):
    weights.append(model.layers[i].get_weights()[0])
    biases.append(model.layers[i].get_weights()[1])
weights = np.asarray(weights)
biases = np.asarray(biases)

os.chdir(r'D:\UrQMD_cent_sim\200\MLFit')
model.save("PaperFitRELU1ModelUrQMD.h5")

for i in range(len(weights)):
    np.savetxt("PaperFitRELU1WeightsUrQMD%s.txt" %
               i, weights[i], delimiter=',', newline='}, \n {')
for i in range(len(biases)):
    np.savetxt("PaperFitRELU1BiasesUrQMD%s.txt" %
               i, biases[i], delimiter=',', newline=",")

# plot_model(
#    model, to_file=r'D:\27gev_production\data\model_imageUrQMD.png', show_shapes=True)

plt.hist2d(targetTest, predictions[-indexT:], bins=[200, 200],
           cmin=0.1, cmap=plt.cm.get_cmap("jet"), norm=LogNorm())

# joint_kws = dict(gridsize=250)

# sns.jointplot(predictions, b,
#              # kind="hex",
#              kind='kde',
#              height=8,
#              ratio=10,
#              space=0,
#              # color='r',
#              cmap='jet',
#              joint_kws=joint_kws
#              )

plt.title("FwdRELUFit vs b: 200 GeV UrQMD, Bias - Test", fontsize=32)
plt.xlabel("RefMult", fontsize=20)
plt.ylabel("FwdRELUFit", fontsize=20)
plt.colorbar()

plt.show()
