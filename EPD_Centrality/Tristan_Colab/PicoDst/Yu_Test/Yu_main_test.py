# \author Skipper Kagamaster
# \date 03/20/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
#

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
import pico_reader as pr
import pico_reader_test as prt
from scipy.stats import skew, kurtosis
import pandas as pd
from joblib import Parallel, delayed

# Enter the number of processor cores you want to use here (cannot exceed physical processors):
cores = 12
run_list = np.loadtxt(r'C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\runs.txt', delimiter=',').astype('int')
sig = np.loadtxt(r'C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\nsigmaVals.txt', delimiter=',')

# Directory where your picos live and where you want to save stuff.
dataDirect = r'E:\2019Picos\14p5GeV\Runs'
saveDirect = r"C:\Users\dansk\Documents\Thesis\Protons\2021_Analysis_Python"
# Loop over the picos.
os.chdir(dataDirect)
r = len(os.listdir())
r_ = 1300  # For loop cutoff (to test on later picos).
count = 1
files = os.listdir()[40:60]
"""
file_len = len(files)
fl_low = int(file_len/cores)
fl = file_len/cores
fl_high = int(np.ceil(file_len/cores))
diff = int(np.round((fl - fl_low)*cores))
if diff != 0:
    files = np.reshape(np.pad(files, (0, cores-diff), 'constant', constant_values=(0, 'taco')), (fl_high, cores))
else:
    files = np.reshape(files, (fl_high, cores))

par_time = time.perf_counter()
for i in files:
    Parallel(n_jobs=-1)(delayed(prt.read_pico)(file, saveDirect) for file in i)
print("Parallel time:", time.perf_counter()-par_time)
"""

seq_time = time.perf_counter()
for i in files:
    if i == 'taco':
        continue
    # Import data from the pico.
    try:
        prt.read_pico(i, saveDirect)
    except Exception as e:  # For any issues that might pop up.
        print("Error on ", i)
        print(e)
        count += 1
        continue
    count += 1
print("Sequential time:", time.perf_counter()-seq_time)
