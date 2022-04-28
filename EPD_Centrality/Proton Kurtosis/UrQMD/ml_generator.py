# \brief UrQMD ML fit generator, including linear fits.
#
#
# \author Skipper Kagamaster
# \date 09/22/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
#

import ml_functions as mlf
import functions as fn
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import uproot as up

"""
The purpose of this bit of code is to take UrQMD generated arrays of
centrality stand ins (reference multiplicity, EPD nMIP sums, EPD
multiplicity ring sums, etc) and generate various fits via machine
learning (ML) methods. 
"""
energy = 7

gev = 7.7
if energy == 200:
    gev = 200
elif energy == 27:
    gev = 27.7
elif energy == 19:
    gev = 19.6
elif energy == 15:
    gev = 14.5

# Load our pandas dataframe with the relevant information and sort it.
# df = pd.read_pickle(r"C:\Users\dansk\Documents\Thesis\2021_DNP_Work\urqmd_{}.pkl".format(energy))
file_loc = r'C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results\BigList\\'
df = pd.read_pickle(file_loc+"out_all.pkl")
df = df[:int(1e6)]
df = df[df['refmult'] > 0]
# refmult3 = df['refmult3'].to_numpy()
refmult3 = df['refmult'].to_numpy()
# b = df['b'].to_numpy()
nprotons = df['net_protons'].to_numpy()
rings = []
for i in range(16):
    rings.append(df["ring{}".format(i+1)].to_numpy())
rings = np.array(rings).T
target = refmult3
target_str = "RefMult3"

# data = up.open(r"D:\UrQMD_cent_sim\{}\CentralityNtuple.root".format(energy))['Rings']
"""
b = data['b'].array(library='np')
refmult3 = data['RefMult3'].array(library='np')
rings = []
for i in range(16):
    rings.append(data['r%02d' % (i+1)].array(library='np'))
rings = np.asarray(rings).T
"""
actFunc = ['linear', 'relu', 'swish', 'mish']
predictions = []
for i in actFunc:
    predictions.append(mlf.ml_run(rings, target, actFunc=i, epochs=100, batch_size=300,
                                  file_loc=file_loc,
                                  energy=energy, min_delta=0.01, patience=8,
                                  loss='mse', optimizer='Nadam', CNN=False))
predictions = np.array(predictions)
for i in range(len(actFunc)):
    print("ActFunc =", actFunc[i])
    for j in range(15):
        print('Result for trial %d => %.1f (expected %.1f)' %
              (j, predictions[i][j], target[j]))

np.save(file_loc+"predictions_{}_refmult.npy".format(energy), predictions)

fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
m_ref = int(np.max(refmult3)*1.2)
m_epd = np.max(predictions, axis=1)*1.2
for i in range(2):
    for j in range(2):
        x = 2*i + j
        count, binsX, binsY = np.histogram2d(target, predictions[x],
                                             bins=m_ref, range=((0, m_ref), (0, m_epd[x])))
        X, Y = np.meshgrid(binsX[:-1], binsY[:-1])
        im = ax[i, j].pcolormesh(X, Y, count, norm=LogNorm(), cmap="jet", shading="auto")
        ax[i, j].set_title(actFunc[x])
        ax[i, j].set_xlabel(target_str)
        ax[i, j].set_ylabel(r"$X_{\zeta',W}$")
        fig.colorbar(im, ax=ax[i, j])
fig.suptitle((r"$\sqrt{s_{NN}}$" + "= {} GeV".format(gev)), fontsize=30)
plt.show()
