# \author Skipper Kagamaster
# \date 10/20/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
#

"""
This code is made to run over UrQMD simulation data and
put it into a Pythonic format for analysis elsewhere. The
analyses supported in this current form are for making ML
predictions of b or RefMult using EPD range inputs and for
doing net proton cumulant analysis with various centrality
metrics (RefMult3, RefMult1, EPD ML mults, and b).
"""

# Not all are in use at present.
import sys
import os
import typing
import logging
import numpy as np
import awkward as ak
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import reader as rdr
from scipy.stats import skew, kurtosis
import pandas as pd
from matplotlib.colors import LogNorm
from keras.models import load_model


data_directory = r"F:\UrQMD\14"
data_directory1 = r"C:\Users\dansk\Documents\Thesis\Protons\2022_urqmd\ucc\picos"
data_directory_1 = r"D:\newfile\15\RACF"
old = False
os.chdir(data_directory)
get_rings = False  # If you want the array to do ML on
get_prots = True  # If you want the histograms to calculate cumulants from
use_ML = True  # If you want to use ML fits in your proton histogramming
target = 'refmult3'  # b, refmult, or refmult3, depending on ML you want to load
just_events = False  # This is to just get the event count from a data set

"""
For the UrQMD files, the available variables are:
refmult, b, x, y, z, pid, p_x, p_y, p_z, p_t, p_g, e, eta, q, rap, m, pr_id
Right now, p_g == e for some reason, and I need to get these in pandas.

UrQMD IDs as follows:
p/n = 1, aparticles are -1
pi = 101
k = 106
Further particle differentiation can be found with charge sign. 
"""

event_total = 0

# Histograms for refmults vs protons.
ref3_bin = 700
ref1_bin = 700
b_bin = 700
b_range = (0, 16.0)
pro_bin = 55
pro_range = (-5, 50)

ref3_range = (0, ref3_bin)
ref1_range = (0, ref1_bin)
ref_bins = [ref3_bin, ref1_bin, b_bin]
ref_ranges = [ref3_range, ref1_range, b_range]

# Arrays to hold ring/b data for ML fits.
ring = [[], [], []]
ring_names = ['b', 'refmult', 'refmult_full']
for i in range(32):
    ring.append([])
    ring_names.append('ring{}'.format(i + 1))

# Histograms to do cumulant analysis on.
refmult_types = ['refmult', 'refmult_full', 'b',
                 'epd_linear', 'epd_relu', 'epd_swish',
                 'epd_CNN']
if use_ML is False:
    refmult_types = ['refmult', 'refmult_full', 'b']
    model_linear = 0
    model_relu = 0
    model_swish = 0
    model_CNN = 0
else:
    model_linear = \
        load_model(r'C:\Users\dansk\Documents\Thesis\Protons\2022_urqmd\linear_{}.h5'.format(target))
    model_relu = \
        load_model(r'C:\Users\dansk\Documents\Thesis\Protons\2022_urqmd\relu_{}.h5'.format(target))
    model_swish = \
        load_model(r'C:\Users\dansk\Documents\Thesis\Protons\2022_urqmd\Swish_{}.h5'.format(target))
    model_CNN = \
        load_model(r'C:\Users\dansk\Documents\Thesis\Protons\2022_urqmd\CNN_{}.h5'.format(target))
    if target == 'refmult3':
        for i in range(len(refmult_types)-3):
            ref_bins.append(ref3_bin)
            ref_ranges.append(ref3_range)
    if target == 'refmult1':
        for i in range(len(refmult_types)-3):
            ref_bins.append(ref1_bin)
            ref_ranges.append(ref1_range)
    else:
        for i in range(len(refmult_types)-3):
            ref_bins.append(b_bin)
            ref_ranges.append(b_range)
prot_types = ['protons', 'antiprotons', 'net protons']
pro_counts = []
pro_bins = []
for i in range(3):
    pro_counts.append([])
    pro_bins.append([])
    for j in range(len(refmult_types)):
        counts, bins_x, bins_y = (np.histogram2d(np.zeros(0),
                                                 np.zeros(0),
                                                 bins=(ref_bins[j], pro_bin),
                                                 range=(ref_ranges[j], pro_range)))
        pro_counts[i].append(counts)
        pro_bins[i].append((bins_x, bins_y))
pro_counts = np.asarray(pro_counts, dtype='object')
pro_bins = np.asarray(pro_bins, dtype='object')

"""**********************************************************
*************************************************************
This will delete all files that were marked as bad if True.
Tread carefully; if there's a code bug in the loop:
***********THIS COULD DELTE ALL THE FILES!!!!!!**************
*************************************************************
"""
del_files = False
"""
*************************************************************
**********************************************************"""

# Loop over the files.
files = []
badfiles = []
for i in np.sort(os.listdir()):
    if i.endswith('root'):
        files.append(i)
f_len = len(files)
# f_len = 5  # For testing.
event_cutoff = 1e9  # If you only want a certain limit on events.
print("Working on file:")
for i in range(f_len):
    if i % 500 == 0:
        print(i + 1, "of", f_len)
    try:
        urqmd = rdr.UrQMD()
        if just_events is True:
            urqmd.import_data_arr(files[i], old=old)
            event_total += len(urqmd.b)
        else:
            if old is True:
                urqmd.import_data_df_old_2(model_linear, model_relu, model_swish, model_CNN, files[i], ML=use_ML)
            else:
                urqmd.import_data_df(model_linear, model_relu, model_swish, model_CNN, files[i], ML=use_ML)
            event_total += len(urqmd.event['b'])
        if get_prots is True:
            if old is True:
                protons, aprotons, nprotons, index = rdr.get_proton_hist_old(urqmd.track)
            else:
                protons, aprotons, nprotons, index = rdr.get_proton_hist(urqmd.track)
            prots = np.asarray((protons, aprotons, nprotons))
            for j in range(3):
                for k in range(len(refmult_types)):
                    pro_counts[j][k] += np.histogram2d(urqmd.event[refmult_types[k]][index],
                                                       prots[j],
                                                       bins=(ref_bins[k], pro_bin),
                                                       range=(ref_ranges[k], pro_range))[0]
        if get_rings is True:
            for j in range(35):
                ring[j] = np.hstack((ring[j], urqmd.event[ring_names[j]].to_numpy()))
        if event_total > event_cutoff:
            break
        # The below are for if you want to make giant, pandas dataframes. Do not recommend.
        # event = pd.concat([event, urqmd.event], ignore_index=True)
        # track = pd.concat([track, urqmd.track])
    except Exception as e:
        print("Error:", e)
        badfiles.append(files[i])
        continue

if len(badfiles) > 0:
    print("Here are the files that failed:")
    print('***********************************')
    print(badfiles, '\n', '\n', '\n')


tot_len = len(badfiles)
if del_files is True:
    for file in badfiles:
        os.remove(file)
    print(tot_len, "files deleted.")
print("Total number of events:", event_total)

# Loop over the files for the ucc events.
os.chdir(data_directory1)
files = []
badfiles = []
for i in np.sort(os.listdir()):
    if i.endswith('root'):
        files.append(i)
f_len = len(files)
# f_len = 5  # For testing.
# event_cutoff = 1e9  # If you only want a certain limit on events.
print("UCC file set.\nWorking on file:")
for i in range(f_len):
    if i % 500 == 0:
        print(i + 1, "of", f_len)
    try:
        urqmd = rdr.UrQMD()
        if just_events is True:
            urqmd.import_data_arr(files[i], old=old)
            event_total += len(urqmd.b)
            if len(urqmd.b) == 0:
                badfiles.append(files[i])
        else:
            if old is True:
                urqmd.import_data_df_old_2(model_linear, model_relu, model_swish, model_CNN, files[i], ML=use_ML)
            else:
                urqmd.import_data_df(model_linear, model_relu, model_swish, model_CNN, files[i], ML=use_ML)
            event_total += len(urqmd.event['b'])
        if get_prots is True:
            if old is True:
                protons, aprotons, nprotons, index = rdr.get_proton_hist_old(urqmd.track)
            else:
                protons, aprotons, nprotons, index = rdr.get_proton_hist(urqmd.track)
            prots = np.asarray((protons, aprotons, nprotons))
            for j in range(3):
                for k in range(len(refmult_types)):
                    pro_counts[j][k] += np.histogram2d(urqmd.event[refmult_types[k]][index],
                                                       prots[j],
                                                       bins=(ref_bins[k], pro_bin),
                                                       range=(ref_ranges[k], pro_range))[0]
        if get_rings is True:
            for j in range(35):
                ring[j] = np.hstack((ring[j], urqmd.event[ring_names[j]].to_numpy()))
        if event_total > event_cutoff:
            break
        # The below are for if you want to make giant, pandas dataframes. Do not recommend.
        # event = pd.concat([event, urqmd.event], ignore_index=True)
        # track = pd.concat([track, urqmd.track])
    except Exception as e:
        print("Error:", e)
        badfiles.append(files[i])
        continue

if len(badfiles) > 0:
    print("Here are the files that failed:")
    print('***********************************')
    print(badfiles, '\n', '\n', '\n')


tot_len = len(badfiles)
if del_files is True:
    for file in badfiles:
        os.remove(file)
    print(tot_len, "files deleted.")
print("Total number of events:", event_total)

# Loop over the files for old, RACF events.
os.chdir(data_directory_1)
files = []
badfiles = []
for i in np.sort(os.listdir()):
    if i.endswith('root'):
        files.append(i)
f_len = len(files)
# f_len = 5  # For testing.
# event_cutoff = 1e9  # If you only want a certain limit on events.
print("RACF file set.\nWorking on file:")
old = True
for i in range(f_len):
    if i % 500 == 0:
        print(i + 1, "of", f_len)
    try:
        urqmd = rdr.UrQMD()
        if just_events is True:
            urqmd.import_data_arr(files[i], old=old)
            event_total += len(urqmd.b)
        else:
            if old is True:
                urqmd.import_data_df_old(model_linear, model_relu, model_swish, model_CNN, files[i], ML=use_ML)
            else:
                urqmd.import_data_df(model_linear, model_relu, model_swish, model_CNN, files[i], ML=use_ML)
            event_total += len(urqmd.event['b'])
        if get_prots is True:
            if old is True:
                protons, aprotons, nprotons, index = rdr.get_proton_hist_old(urqmd.track)
            else:
                protons, aprotons, nprotons, index = rdr.get_proton_hist(urqmd.track)
            prots = np.asarray((protons, aprotons, nprotons))
            for j in range(3):
                for k in range(len(refmult_types)):
                    pro_counts[j][k] += np.histogram2d(urqmd.event[refmult_types[k]][index],
                                                       prots[j],
                                                       bins=(ref_bins[k], pro_bin),
                                                       range=(ref_ranges[k], pro_range))[0]
        if get_rings is True:
            for j in range(35):
                ring[j] = np.hstack((ring[j], urqmd.event[ring_names[j]].to_numpy()))
        if event_total > event_cutoff:
            break
        # The below are for if you want to make giant, pandas dataframes. Do not recommend.
        # event = pd.concat([event, urqmd.event], ignore_index=True)
        # track = pd.concat([track, urqmd.track])
    except Exception as e:
        print("Error:", e)
        badfiles.append(files[i])
        continue

if len(badfiles) > 0:
    print("Here are the files that failed:")
    print('***********************************')
    print(badfiles, '\n', '\n', '\n')


tot_len = len(badfiles)
if del_files is True:
    for file in badfiles:
        os.remove(file)
    print(tot_len, "files deleted.")

print("We're out of the loops.")

print("Total number of events:", event_total)
print("Total failed files:", tot_len)

os.chdir(data_directory)

if get_rings is True:
    ring = np.asarray(ring)
    np.save('ring.npy', ring)

if get_prots is True:
    np.save('pro_counts.npy', pro_counts)
    np.save('pro_bins.npy', pro_bins)
    fig, ax = plt.subplots(3, 3, figsize=(16, 9), constrained_layout=True)
    for i in range(3):
        for j in range(3):
            x = j+4
            im = ax[i, j].pcolormesh(pro_bins[i][x][1].astype('float'),
                                     pro_bins[i][x][0].astype('float'),
                                     pro_counts[i][x].astype('float'),
                                     cmap='jet', norm=LogNorm())
            ax[i, j].set_ylabel(refmult_types[x])
            ax[i, j].set_xlabel(prot_types[i])
            fig.colorbar(im, ax=ax[i, j])
    plt.show()
