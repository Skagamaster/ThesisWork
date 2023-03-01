#
# \Analyse our proton arrays and maybe make a dataframe.
#
# \author Skipper Kagamaster
# \date 06/02/2022
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
# /

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# badruns = np.load(r"C:\200\PythonArrays\badruns_protons.npy", allow_pickle=True)
# badruns = np.char.add(badruns.astype('str'), "_protons.npy")
badruns = np.load(r"C:\200\PythonArrays\badruns.npy", allow_pickle=True)
os.chdir(r"C:\200\PythonArrays\Analysis_Proton_Arrays")
files = np.asarray(os.listdir(), dtype='str')
files = files[np.isin(files, badruns, invert=True)]
refmult3 = []
protons = []
antiprotons = []
rings = []
for i in range(32):
    rings.append([])

count = 0
print("Working on:")
for i in files:
    if count % 100 == 0:
        print(count, "of", len(files))
    count += 1
    if i.endswith('.npy'):
        arr = np.load(i, allow_pickle=True)
        refmult3 = np.hstack((refmult3, arr[2]))
        protons = np.hstack((protons, arr[0]))
        antiprotons = np.hstack((antiprotons, arr[1]))
        for j in range(32):
            rings[j] = np.hstack((rings[j], arr[j + 3]))

df = pd.DataFrame({'RefMult3': refmult3})
df['protons'] = protons
df['antiprotons'] = antiprotons
for i in range(32):
    df['ring{}'.format(i+1)] = rings[i]
print(df)
df.to_pickle('full_set.pkl')
