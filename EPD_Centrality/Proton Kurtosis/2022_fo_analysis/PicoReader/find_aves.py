#
# \Find mean and variance for quantities in picos.
#
# \author Skipper Kagamaster
# \date 05/03/2022
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
# /

"""
This code is to get a few things you'll need in order to
start the analysis. You'll need average quantities in order
to find your bad runs (processed in Read_Aves.py). But first,
you need to calibration values for the EPD in order to have
correct values. This functionality is included if calibrations
are complete.
In addition, you'll want some histograms for both display
purposes (to see if everything works as it ought) and to
calibrate nSigmaProton mean shift (done in nSigmaCalibrate.py).
"""

import uproot as up
import awkward as ak
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pico_reader as rdr

data_direct = r'F:\2019Picos\14p5GeV\Runs'  # Directory where the picos live
save_direct = r'D:\14GeV\Thesis\PythonArrays'  # Directory for saving results

os.chdir(data_direct)
files = []
for i in os.listdir():
    if i.endswith('root'):
        files.append(i)
"""
These are to load the files for 14.5 GeV nMIP calibration for the EPD in FastOffline.
"""
cal_set = np.asarray([94, 105, 110, 113, 114, 123, 138, 139])
cal_files = []
for i in cal_set:
    cal_files.append(np.loadtxt(r'D:\14GeV\ChiFit\adc{}.txt'.format(i), delimiter=','))

"""
These are to hold pileup correlations for later cuts.
"""
rt_count, rt_binsX, rt_binsY = np.histogram2d([0], [0], bins=(1000, 1700),
                                              range=((0, 1000), (0, 1700)))
rm_count, rm_binsX, rm_binsY = np.histogram2d([0], [0], bins=1000,
                                              range=((0, 1000), (0, 500)))
rb_count, rb_binsX, rb_binsY = np.histogram2d([0], [0], bins=(1000, 400),
                                              range=((0, 1000), (0, 400)))

aves = []  # Holds average data run by run
stds = []  # Holds std data run by run
runs = [[], [], []]  # Holds runID, event#, track# (in case needed for statistics)
running_counter = 0

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
        # Pileup histogram filling
        rt_count += np.histogram2d(pico.refmult3, pico.tofmult,
                                   bins=(1000, 1700), range=((0, 1000), (0, 1700)))[0]
        rm_count += np.histogram2d(pico.refmult3, pico.tofmatch,
                                   bins=1000, range=((0, 1000), (0, 500)))[0]
        rb_count += np.histogram2d(pico.refmult3, pico.beta_eta_1,
                                   bins=(1000, 400), range=((0, 1000), (0, 400)))[0]
        nSigmaProton = []  # For nSigmaProton mean shift calibrations
        nsig = ak.to_numpy(ak.flatten(pico.nsigma_proton[index][index_t]))
        p_flat = ak.to_numpy(ak.flatten(pico.p_g[index][index_t]))
        for j in range(13):
            count, bins = np.histogram(nsig[(p_flat > 0.1 * j + 0.1) & (p_flat <= 0.1 * j + 0.2)],
                                       bins=400, range=(-10, 10))
            nSigmaProton.append(count)
        nSigmaProton = np.asarray(nSigmaProton)
        np.save(r'D:\14GeV\Thesis\PythonArrays\nSigmaProtonCal\{}_nSigRaw.npy'.format(pico.run_id),
                nSigmaProton)
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
np.save('ref_tof_mult.npy', rt_count)
np.save('ref_tof_mult_bins.npy', np.asarray([rt_binsX, rt_binsY], dtype='object'))
np.save('ref_tof_match.npy', rm_count)
np.save('ref_tof_match_bins.npy', np.asarray([rm_binsX, rm_binsY], dtype='object'))
np.save('ref_beta.npy', rb_count)
np.save('ref_beta_bins.npy', np.asarray([rb_binsX, rb_binsY], dtype='object'))
