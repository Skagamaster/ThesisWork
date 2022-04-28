# \Display QA quantities from FastOffline data
#
#
# \author Skipper Kagamaster
# \date 04/28/2022
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
#

import uproot as up
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import pandas as pd

os.chdir(r'D:\14Gev\Thesis\After_QA_Picos')
files = []
for i in os.listdir():
    if i.endswith('root'):
        files.append(i)

keys = ['v_z', 'refmult', 'v_r', 'rt_mult', 'rt_matchd', 'ref_beta', 'mpq', 'betapd',
        'p_t', 'phi', 'dca', 'eta', 'rap', 'nhitsq', 'nhits_dedx', 'nhitsfit_ratio', 'dedx_pq',
        'av_z', 'arefmult', 'av_r', 'art_mult', 'art_match', 'aref_beta', 'ampq', 'abetap',
        'ap_t', 'aphi', 'adca', 'aeta', 'arap', 'anhitsq', 'anhits_dedx', 'anhitsfit_ratio', 'adedx_pq',
        'pdedx_pq', 'p_t_p_g', 'msq',
        'hNSigmaProton_0', 'hNSigmaProton_1', 'hNSigmaProton_2', 'hNSigmaProton_3',
        'hNSigmaProton_4', 'hNSigmaProton_5', 'hNSigmaProton_6', 'hNSigmaProton_7',
        'hNSigmaProton_8', 'hNSigmaProton_9', 'hNSigmaProton_10',
        'Protons;1']
keys = np.asarray(keys)
set_1d = np.array((0, 1, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 25, 26, 27, 28, 29, 30, 31, 32, 36,),
                  dtype='int')
set_2d = np.array((2, 3, 4, 5, 6, 7, 16, 19, 20, 21, 22, 23, 24, 33, 34, 35), dtype='int')
data = up.open(files[0])
df_1d = pd.DataFrame()
df_2d = pd.DataFrame()
for i in keys[set_1d]:
    df_1d[i] = data[i].to_numpy()
for i in keys[set_2d]:
    df_2d[i] = data[i].to_numpy()
for i in files[1:30]:
    data = up.open(i)
    for j in keys[set_1d]:
        df_1d[j][0] += data[j].to_numpy()[0]
    for j in keys[set_2d]:
        df_2d[j][0] += data[j].to_numpy()[0]

X, Y = np.meshgrid(df_2d['dedx_pq'][2], df_2d['dedx_pq'][1])
plt.pcolormesh(Y, X, df_2d['dedx_pq'][0], cmap='bone', norm=LogNorm())
plt.pcolormesh(Y, X, df_2d['pdedx_pq'][0], cmap='jet', norm=LogNorm())
plt.colorbar()
plt.show()
