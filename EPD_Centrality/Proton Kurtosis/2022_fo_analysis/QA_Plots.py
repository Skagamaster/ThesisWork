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
badruns = np.load(r'D:\14Gev\Thesis\badruns.npy', allow_pickle=True)
morebadruns = np.load(r'D:\14Gev\Thesis\more_badruns.npy', allow_pickle=True)
badruns = np.unique(np.hstack((badruns, morebadruns))).astype('int')
goodruns = np.load(r'D:\14GeV\Thesis\test_good_runs.npy', allow_pickle=True)
print(badruns)
files = []
for i in os.listdir():
    if int(i[4:12]) not in goodruns:
        print(i, "is a very bad run. Bad run!")
        continue
    if int(i[4:12]) < 20120000:
        pass
    if i.endswith('root'):
        files.append(i)

keys = np.array(['v_z', 'refmult', 'v_r', 'rt_mult', 'rt_matchd', 'ref_beta', 'mpq', 'betapd',
                 'p_t', 'phi', 'dca', 'eta', 'rap', 'nhitsq', 'nhits_dedx', 'nhitsfit_ratio', 'dedx_pq',
                 'av_z', 'arefmult', 'av_r', 'art_mult', 'art_match', 'aref_beta', 'ampq', 'abetap',
                 'ap_t', 'aphi', 'adca', 'aeta', 'arap', 'anhitsq', 'anhits_dedx', 'anhitsfit_ratio',
                 'adedx_pq', 'pdedx_pq', 'p_t_p_g', 'msq',
                 'hNSigmaProton_0', 'hNSigmaProton_1', 'hNSigmaProton_2', 'hNSigmaProton_3',
                 'hNSigmaProton_4', 'hNSigmaProton_5', 'hNSigmaProton_6', 'hNSigmaProton_7',
                 'hNSigmaProton_8', 'hNSigmaProton_9', 'hNSigmaProton_10',
                 'Protons;1'])
pro_keys = np.array(['RefMult3', 'protons', 'antiprotons', 'net_protons',
                     'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9',
                     'r10', 'r11', 'r12', 'r13', 'r14', 'r15', 'r16', 'r17',
                     'r18', 'r19', 'r20', 'r21', 'r22', 'r23', 'r24', 'r25',
                     'r26', 'r27', 'r28', 'r29', 'r30', 'r31', 'r32'])
set_1d = np.array((0, 1, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 25, 26, 27, 28, 29, 30, 31, 32, 36,),
                  dtype='int')
set_2d = np.array((2, 3, 4, 5, 6, 7, 16, 19, 20, 21, 22, 23, 24, 33, 34, 35), dtype='int')
data = up.open(files[0])
df_1d = pd.DataFrame()
df_2d = pd.DataFrame()
df_protons = pd.DataFrame()
refmult = [[], [], []]
refmult[2].append(files[0][4:12])
for i in keys[set_1d]:
    df_1d[i] = data[i].to_numpy()
for i in keys[set_2d]:
    df_2d[i] = data[i].to_numpy()
for i in pro_keys:
    df_protons[i] = data['Protons'][i].array(library='np')
refmult[0].append(np.mean(df_protons['RefMult3'].to_numpy()))
refmult[1].append(np.var(df_protons['RefMult3'].to_numpy()))
count = 1
print("Working on file:")
for i in files:
    new_df = pd.DataFrame()
    if count % 50 == 0:
        print(count, "of", len(files))
    data = up.open(i)
    for j in keys[set_1d]:
        df_1d[j][0] += data[j].to_numpy()[0]
    for j in keys[set_2d]:
        df_2d[j][0] += data[j].to_numpy()[0]
    for j in pro_keys:
        new_df[j] = data['Protons'][j].array(library='np')
    refmult[0].append(np.mean(new_df['RefMult3'].to_numpy()))
    refmult[1].append(np.var(new_df['RefMult3'].to_numpy()))
    refmult[2].append(i[4:12])
    df_protons = pd.concat((df_protons, new_df), ignore_index=True)
    count += 1
df_1d.to_pickle(r'D:\14Gev\Thesis\QA_df_1d.pkl')
df_2d.to_pickle(r'D:\14Gev\Thesis\QA_df_2d.pkl')
df_protons.to_pickle(r'D:\14GeV\Thesis\df_protons.pkl')

refmult = np.asarray(refmult)
np.save(r'D:\14GeV\Thesis\refmult_ave.npy', refmult)

bases = ['v_r', 'rt_mult', 'dedx_pq', 'ref_beta', 'mpq', 'betapd']
cuts = ['av_r', 'art_mult', 'adedx_pq', 'aref_beta', 'ampq', 'abetap']
x_labels = [r'$v_x$ (cm)', 'RefMult3', r'$p \cdot q$ ($ \frac{GeV}{c}$)', 'RefMult3',
            r'$p \cdot q$ ($ \frac{GeV}{c}$)', r'p ($\frac{GeV}{c}$)']
y_labels = [r'$v_y$ (cm)', 'TofMult', r'$\frac{dE}{dx}$ ($\frac{KeV}{cm}$)', r'$\beta\eta1$',
            r'$m^2$ ($\frac{GeV^2}{c^4}$)', r'$\frac{1}{\beta}$']

'''
fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
for i in range(2):
    for j in range(3):
        x = i * 3 + j
        X, Y = np.meshgrid(df_2d[bases[x]][2], df_2d[bases[x]][1])
        im = ax[i, j].pcolormesh(Y, X, df_2d[bases[x]][0], cmap='jet', norm=LogNorm())
        ax[i, j].set_xlabel(x_labels[x], fontsize=12)
        ax[i, j].set_ylabel(y_labels[x], fontsize=12)
        fig.colorbar(im, ax=ax[i, j])
plt.show()

fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
for i in range(2):
    for j in range(3):
        x = i * 3 + j
        X, Y = np.meshgrid(df_2d[bases[x]][2], df_2d[bases[x]][1])
        ax[i, j].pcolormesh(Y, X, df_2d[bases[x]][0], cmap='bone', norm=LogNorm())
        im = ax[i, j].pcolormesh(Y, X, df_2d[cuts[x]][0], cmap='jet', norm=LogNorm())
        ax[i, j].set_xlabel(x_labels[x], fontsize=12)
        ax[i, j].set_ylabel(y_labels[x], fontsize=12)
        fig.colorbar(im, ax=ax[i, j])
plt.show()
'''
cuts = ['av_r', 'art_mult', 'pdedx_pq', 'aref_beta', 'ampq', 'abetap']
fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
for i in range(2):
    for j in range(3):
        x = i * 3 + j
        X, Y = np.meshgrid(df_2d[bases[x]][2], df_2d[bases[x]][1])
        ax[i, j].pcolormesh(Y, X, df_2d[bases[x]][0], cmap='bone', norm=LogNorm())
        im = ax[i, j].pcolormesh(Y, X, df_2d[cuts[x]][0], cmap='jet', norm=LogNorm())
        ax[i, j].set_xlabel(x_labels[x], fontsize=12)
        ax[i, j].set_ylabel(y_labels[x], fontsize=12)
        fig.colorbar(im, ax=ax[i, j])
plt.show()
