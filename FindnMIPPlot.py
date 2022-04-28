import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm
import uproot as up
import os
from matplotlib.colors import LogNorm
from numba import vectorize

os.chdir(r'D:\19GeV\Picos')
Day88 = np.load('Day88array.npy', allow_pickle=True)
binNum = max(Day88[0][0][0])-min(Day88[0][0][0])+1
plt.hist(Day88[0][0][0], bins=binNum, histtype='step')
plt.xlim(30, 500)
plt.ylim(0, 2500)
plt.show()
