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

data_direct = r'D:\14GeV\Full_Picos'  # Directory where the picos live
save_direct = r'D:\14GeV\Thesis\PythonArrays'  # Directory for saving results

'''
Let's start by testing this out on a single pico and see if what we got
by using Welford's Algorithm in ROOT was a success.
'''
os.chdir(data_direct)
file = '20094051.picoDst.root'
pico = rdr.PicoDST()
pico.import_data(file)
'''
Now to make the same event cuts; namely:
|v_z| <= 30 cm,
v_r < 2 cm (this could be 1 cm; see the ROOT macro!)
|v_x, v_y, v_z| >= 1e-5 cm
TofMult <= 2.536*RefMult3 + 200.00
TofMult >= 1.352*RefMult3 - 54.08
'''
index = ((abs(pico.v_z) <= 30) & (pico.v_r < 2) & (abs(pico.v_x) >= 1.0e-5) &
         (abs(pico.v_y) >= 1.0e-5) & (abs(pico.v_z) >= 1.0e-5) &
         (pico.tofmult <= (2.536*pico.refmult3 + 200.0)) &
         (pico.tofmult <= (1.352*pico.refmult3 - 54.08)))
plt.hist(pico.v_r, bins=300, histtype='step')
plt.show()
plt.hist2d(pico.v_x, pico.v_y, bins=300, cmin=1)
plt.show()
refmult3 = pico.refmult3[index]
v_z = pico.v_z[index]
print(len(pico.refmult3), len(refmult3), np.mean(refmult3), np.mean(v_z))
