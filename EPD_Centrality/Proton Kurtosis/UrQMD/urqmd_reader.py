# \brief UrQMD reader for use with net proton kurtosis
#        analysis (using arrays already generated).
#
# \author Skipper Kagamaster
# \date 09/20/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
#

import functions as fn
import os
import pandas as pd
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

# Location of UrQMD files
file_loc = r"D:\newfile\200"

"""
The class UrQMD has the following objects:
b = distributions of impact parameters
mult = multiplicity distribution
pid = particle ID (track distribution; awkward array)
pt = particle transverse momentum (track distribution; awkward array)
eta = particle pseudorapidity (track distribution; awkward array)
phi = particle azimuthal angle (track distribution; awkward array)
refmult = all p+-, pi+-, and k+- in |eta| < 1.0
refmult3 = all pi+- and k+- in |eta| < 1.0
epd_rings = D16 array for eventwise sums in all EPD rings (p+-, pi+-, k+-)
protons = all protons in |eta| <= 0.5 and 0.4 <= pt <= 2.0
aprotons = all antiprotons in |eta| <= 0.5 and 0.4 <= pt <= 2.0
nprotons = all net protons in |eta| <= 0.5 and 0.4 <= pt <= 2.0
"""

refmult = np.empty(0)
refmult3 = np.empty(0)
epd_rings = np.empty((16, 0))
protons = np.empty(0)
aprotons = np.empty(0)
nprotons = np.empty(0)
b = np.empty(0)

os.chdir(file_loc)
files = []
print("Working on:")
count = 1
total = len(os.listdir())
for i in os.listdir():
    if count % 10 == 0:
        print(count, "of", total)
    if i.startswith("out"):
        pico = fn.UrQMD()
        pico.get_data(file=i)
        refmult = np.hstack((refmult, pico.refmult))
        refmult3 = np.hstack((refmult3, pico.refmult3))
        epd_rings = np.hstack((epd_rings, pico.epd_rings))
        protons = np.hstack((protons, pico.protons))
        aprotons = np.hstack((aprotons, pico.aprotons))
        nprotons = np.hstack((nprotons, pico.nprotons))
        b = np.hstack((b, pico.b))
    count += 1
    if count >= 3:
        pass

df = pd.DataFrame({"b": b, "refmult": refmult, "refmult3": refmult3, "protons": protons,
                   "antiprotons": aprotons, "net_protons": nprotons})
for i in range(16):
    df["ring{}".format(i+1)] = epd_rings[i]
print(df)

os.chdir(r"C:\Users\dansk\Documents\Thesis\2021_DNP_Work")
df.to_pickle("urqmd_200.pkl")

df['b'].plot.hist(bins=100, range=(0, 17), histtype='step')
plt.show()
