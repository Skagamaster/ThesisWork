#
# \Find mean and variance for quantities in picos.
#
# \author Skipper Kagamaster
# \date 05/03/2022
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
# /

import uproot as up
import awkward as ak
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pico_reader as rdr

data_direct = r'F:\2019Picos\14p5GeV\Runs'  # Directory where the picos live
save_direct = r'D:\14GeV\Thesis\PythonArrays'  # Directory for saving results

'''
Let's start by testing this out on a single pico and see if what we got
by using Welford's Algorithm in ROOT was a success.
'''
os.chdir(data_direct)
files = []
for i in os.listdir():
    if i.endswith('root'):
        files.append(i)

'''
These are to load the files for 14.5 GeV nMIP calibration for the EPD in FastOffline.
'''
cal_set = np.asarray([94, 105, 110, 113, 114, 123, 138, 139])
cal_files = []
for i in cal_set:
    cal_files.append(np.loadtxt(r'D:\14GeV\ChiFit\adc{}.txt'.format(i), delimiter=','))

aves = []  # Holds average data run by run
stds = []  # Holds std data run by run
runs = [[], [], []]  # Holds runID, event#, track# (in case needed for statistics)
running_counter = 0
# Let's make a Pandas dataframe.
columns = ['<RefMult3>', '<v_z>', '<v_r>', 'zdcx']
for i in range(32):
    columns.append('<ring_{}>'.format(i+1))

print("Chewing on file:")
for i in files:
    try:
        if running_counter % 10 == 0:
            print(running_counter + 1, "of", len(files))
        running_counter += 1
        day = int(i[2:5])
        cal_index = np.where(cal_set <= day)[0][-1]
        epd_cal = cal_files[cal_index]
        pico = rdr.PicoDST()
        pico.import_data(i, cal_file=epd_cal)
        index = (abs(pico.v_z) <= 30.0)  # Basic vertex cut
        index_t = abs(pico.nhitsfit[index]) >= 10  # Low level track quality cut
        runs[0].append(pico.run_id)
        runs[1].append(len(pico.refmult3[index]))
        runs[2].append(len(ak.to_numpy(ak.flatten(pico.p_t[index][index_t]))))
        aves.append(np.hstack((np.mean([pico.refmult3[index],
                                        pico.v_z[index],
                                        pico.v_r[index],
                                        pico.zdcx[index],
                                        *pico.epd_hits.T[index].T],
                                       axis=1),
                               np.mean([ak.to_numpy(ak.flatten(pico.p_t[index][index_t])),
                                        ak.to_numpy(ak.flatten(pico.phi[index][index_t])),
                                        ak.to_numpy(ak.flatten(pico.eta[index][index_t])),
                                        ak.to_numpy(ak.flatten(pico.dca[index][index_t]))],
                                       axis=1))))
        stds.append(np.hstack((np.std([pico.refmult3[index],
                                       pico.v_z[index],
                                       pico.v_r[index],
                                       pico.zdcx[index],
                                       *pico.epd_hits.T[index].T],
                                      axis=1),
                               np.std([ak.to_numpy(ak.flatten(pico.p_t[index][index_t])),
                                       ak.to_numpy(ak.flatten(pico.phi[index][index_t])),
                                       ak.to_numpy(ak.flatten(pico.eta[index][index_t])),
                                       ak.to_numpy(ak.flatten(pico.dca[index][index_t]))],
                                      axis=1))))
    except Exception as e:  # For any issues that might pop up.
        print("Error on ", i)
        print(e)
        running_counter += 1
        continue
print("All glory to the hypnotoad!")
runs = np.asarray(runs)
aves = np.asarray(aves)
stds = np.asarray(stds)
os.chdir(save_direct)
np.save('ave_runs.npy', runs)
np.save('ave_aves.npy', aves)
np.save('ave_stds.npy', stds)
