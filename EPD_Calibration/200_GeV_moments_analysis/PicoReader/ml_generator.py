# \brief STAR FastOffline data ML fit generator, including linear fits.
#
#
# \author Skipper Kagamaster
# \date 04/28/2022
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
#

import ml_functions as mlf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import uproot as up

import tensorflow as tf
import random as python_random


def seed_set():
    np.random.seed(10001)
    python_random.seed(20)
    tf.random.set_seed(405)
    return


"""
The purpose of this bit of code is to take UrQMD generated arrays of
centrality stand ins (reference multiplicity, EPD nMIP sums, EPD
multiplicity ring sums, etc) and generate various fits via machine
learning (ML) methods. 
"""

# Inputs:
energy = 200  # The COM energy of your data.
urqmd = False  # Simulation or live data ("True" if simulation)
target_str = 'refmult3'  # Which to target: refmult3 or refmult1
file_loc = r'C:\200\PythonArrays\Analysis_Proton_Arrays'  # Data drive
save_loc = r'C:\200\ML'  # ML folder

"""None of the below should have to be modified, though it can of course
be if you want to make tweaks."""

gev = 200
if energy == 200:
    gev = 200
elif energy == 11:
    gev = 11
elif energy == 27:
    gev = 27.7
elif energy == 19:
    gev = 19.6
elif energy == 14:
    gev = 14.6

"""
First, we'll need to load our datasets. These will be different depending on if you
are doing UrQMD simulation or actual, STAR data.
"""
os.chdir(file_loc)
df = pd.read_pickle('full_set.pkl')
refmult3 = df['RefMult3'].to_numpy()
# If you need truncation.
# refmult3 = refmult3[-2232269:]
rings = []
for i in range(32):
    arr = df['ring{}'.format(i+1)].to_numpy()
    # If you need truncation.
    # arr = arr[-2232269:]
    rings.append(arr)
rings = np.asarray(rings)
target = refmult3
min_delta = 0.1
rings = rings.T
"""
Now for the ML fitting.
"""
actFunc = ['linear', 'relu']
ylabels = [r'$X_{LW}$', r'$X_{relu}$']
predictions = []
model = []

batch_size = int(len(target)/10000)
print(batch_size)
os.chdir(save_loc)

for i in actFunc:
    pred, mod, w, b = mlf.ml_run(rings, target, target_str, actFunc=i, epochs=300, batch_size=batch_size,
                                 min_delta=min_delta, patience=10, loss='logcosh', optimizer='Nadam',
                                 CNN=False, h_lay=2)
    mod.save('{}_{}.h5'.format(i, target_str))
    np.save('{}_{}_weights.npy'.format(i, target_str), w)
    np.save('{}_{}_biases.npy'.format(i, target_str), b)
    predictions.append(pred)

    """This is to use a set seed to avoid getting stuck in a saddle point.
    You can turn it off, but you may get stuck.
    Note: Results will not be random from run to run if you
    use this, so if you're trying to vet results from different
    ML runs then try something else."""
    if i == 'Swish':
        seed_set()

predictions = np.asarray(predictions, dtype='object')
for i in range(len(actFunc)):
    print("ActFunc:", actFunc[i])
    for j in range(15):
        print('Result for trial %d => %.1f (expected %.1f)' %
              (j, predictions[i][j], target[j]))
np.save("predictions_{}_{}.npy".format(energy, target_str), predictions)
df['linear'] = predictions[0]
df['relu'] = predictions[1]
df.to_pickle('full_set.pkl')

fig, ax = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)
m_ref = int(np.max(target)*1.2)
m_epd = np.max(predictions, axis=1)*1.2
for i in range(2):
    count, binsX, binsY = np.histogram2d(target, predictions[i],
                                         bins=200, range=((0, m_ref), (0, m_epd[i])))
    X, Y = np.meshgrid(binsX[:-1], binsY[:-1])
    count[count < 10] = 0
    im = ax[i].pcolormesh(Y, X, count, norm=LogNorm(), cmap="jet", shading="auto")
    ax[i].set_title(actFunc[i])
    ax[i].set_xlabel(target_str + ' (fm)')
    ax[i].set_ylabel(ylabels[i])
    fig.colorbar(im, ax=ax[i])
fig.suptitle((r"$\sqrt{s_{NN}}$" + "= {} GeV".format(gev)), fontsize=30)
plt.show()
