# \brief UrQMD ML fit generator, including linear fits.
#
#
# \author Skipper Kagamaster
# \date 01/03/2022
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
energy = 14  # The COM energy of your data.
urqmd = False  # Simulation or live data ("True" if simulation)
target_str = 'refmult3'  # Which to target: refmult3, refmult1, or b
file_loc = r'F:\UrQMD\14'  # UrQMD drive
file_loc = r'C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results\BigList'  # data drive
file_loc_1 = r'C:\Users\dansk\Documents\Thesis\Protons\2022_urqmd\ucc\picos'  # ucc data drive
save_loc = r'C:\Users\dansk\Documents\Thesis\Protons\2022_urqmd'  # UrQMD save drive
save_loc = r'C:\Users\dansk\Documents\Thesis\Protons\2022_data'  # data save drive

"""None of the below should have to be modified, though it can of course
be if you want to make tweaks."""

gev = 14
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
if urqmd is True:
    ring = np.load('ring.npy', allow_pickle=True)
    b = ring[0]
    refmult3 = ring[1]
    refmult1 = ring[2]
    epd_rings = ring[3:]
    rings = epd_rings[:16]
    rings = np.add(rings, epd_rings[16:])

    os.chdir(file_loc_1)
    ring1 = np.load('ring.npy', allow_pickle=True)
    b1 = ring1[0]
    refmult31 = ring1[1]
    refmult11 = ring1[2]
    epd_rings1 = ring1[3:]
    rings1 = epd_rings1[:16]
    rings1 = np.add(rings1, epd_rings1[16:])

    b = np.hstack((b, b1))
    refmult3 = np.hstack((refmult3, refmult31))
    refmult1 = np.hstack((refmult1, refmult11))
    rings = np.hstack((rings, rings1))

# TODO Add in the data option (this is all a placeholder for now).
else:
    df = pd.read_pickle('out_all.pkl')
    refmult3 = df['refmult'].to_numpy()
    rings = []
    for i in range(32):
        rings.append(df['ring{}'.format(i+1)].to_numpy())
    rings = np.asarray(rings)
if target_str == 'b':
    target = b
    min_delta = 0.01
elif target_str == 'refmult':
    target = refmult1
    min_delta = 0.1
elif target_str == 'refmult3':
    target = refmult3
    min_delta = 0.1
else:
    print("You must pick a valid target.")
    target = None
    min_delta = None
    quit()

fig, ax = plt.subplots(4, 4, figsize=(16, 9), constrained_layout=True)
for i in range(4):
    for j in range(4):
        x = i*4 + j
        if x == 0:
            count, binsX, binsY = np.histogram2d(target, rings[x],
                                                 bins=100)
        elif x == 1:
            count, binsX, binsY = np.histogram2d(target, rings[x],
                                                 bins=60)
        else:
            count, binsX, binsY = np.histogram2d(target, rings[x],
                                                 bins=30)
        X, Y = np.meshgrid(binsX[:-1], binsY[:-1])
        count[count < 40] = 0
        im = ax[i, j].pcolormesh(X, Y, count.T, norm=LogNorm(), cmap="jet", shading="auto")
        ax[i, j].set_xlabel(target_str + ' (fm)')
        ax[i, j].set_ylabel("Ring {}".format(x+1))
        fig.colorbar(im, ax=ax[i, j])
# fig.suptitle((r"$X_{\Sigma}$ vs b, $\sqrt{s_{NN}}$" + "= {} GeV".format(gev)), fontsize=30)
plt.show()

rings = rings.T
"""
Now for the ML fitting.
"""
# TODO Add in unsupervised learning (which will require post-fit processing).
# TODO Figure out why Bose doesn't work.
actFunc = ['linear', 'relu', 'Swish', 'CNN']
ylabels = [r'$X_{LW}$', r'$X_{relu}$', r'$X_{swish}$', r'$X_{CNN}$']
predictions = []
model = []

batch_size = int(len(target)/10000)
print(batch_size)

os.chdir(save_loc)
for i in actFunc:
    pred, mod = mlf.ml_run(rings, target, target_str, actFunc=i, epochs=300, batch_size=batch_size,
                                  min_delta=min_delta, patience=10, loss='logcosh', optimizer='Nadam',
                                  CNN=False, h_lay=2)

    # Just for grins
    """
    pred, mod = mlf.ml_run(rings, target, actFunc=i, epochs=200, batch_size=600,
                              min_delta=0.0005, patience=25, loss='mse', optimizer='Nadam',
                              CNN=False, actFunc2='Swish', h_lay=4)
    """
    mod.save('{}_{}.h5'.format(i, target_str))
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
# TODO Make this into a pandas df for the labels.
np.save("predictions_{}_{}.npy".format(energy, target_str), predictions)

fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
m_ref = int(np.max(target)*1.2)
m_epd = np.max(predictions, axis=1)*1.2
for i in range(2):
    for j in range(2):
        x = i*2 + j
        count, binsX, binsY = np.histogram2d(target, predictions[x],
                                             bins=200, range=((0, m_ref), (0, m_epd[x])))
        X, Y = np.meshgrid(binsX[:-1], binsY[:-1])
        count[count < 10] = 0
        im = ax[i, j].pcolormesh(Y, X, count, norm=LogNorm(), cmap="jet", shading="auto")
        ax[i, j].set_title(actFunc[x])
        ax[i, j].set_xlabel(target_str + ' (fm)')
        ax[i, j].set_ylabel(ylabels[x])
        fig.colorbar(im, ax=ax[i, j])
fig.suptitle((r"$\sqrt{s_{NN}}$" + "= {} GeV".format(gev)), fontsize=30)
plt.show()
