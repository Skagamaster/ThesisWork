import numpy as np
from numpy import loadtxt
import seaborn as sns
import matplotlib.pyplot as plt
import os

# This macro is simply to make some plots for EPD RefMult.
# Actual analysis has been performed elsewhere. Go there. Seriously.

epdRef = np.loadtxt(r'D:\27gev_production\data\DisplayFiles\EPDforBNL\epd.txt')
tpcRef = np.loadtxt(r'D:\27gev_production\data\DisplayFiles\EPDforBNL\tpc.txt')


# This portion is for jointplot KDE or REG.
plt.rcParams["axes.labelsize"] = 15

sns.set(style="dark")
joint_kws = dict(gridsize=250)
h = sns.jointplot(epdRef, tpcRef,
                  # kind="hex",
                  kind="kde",
                  xlim=(-20, 300),
                  ylim=(-20, 300),
                  height=8,
                  ratio=10,
                  space=0,
                  dropna=True,
                  # cmap='hot',
                  cmap='jet',
                  # cmap='Blues',
                  joint_kws=joint_kws
                  # kind='reg',
                  # color='b',
                  # alpha=0.1,
                  # joint_kws={'line_kws': {'color': 'r'}}
                  )

h.fig.suptitle(
    r"RefMult: TPC vs EPD, Au+Au $\sqrt{s}$=27 GeV", verticalalignment='top', fontsize=32)
h.set_axis_labels('EPD refMult', 'TPC refMult', fontsize=20)
'''
# 2D histogram of the whole thing (for display).
plt.hist2d(epdRef, tpcRef,
           bins=[822, 411],
           range=([0, 410], [0, 410]),
           cmin=0.1,
           # vmin=0,
           # vmax=30,
           # cmax=200,
           alpha=1.0,
           cmap=plt.cm.get_cmap("jet"))

plt.title(r"RefMult: TPC vs EPD, Au+Au $\sqrt{s}$=27 GeV", fontsize=32)
plt.xlabel("EPD refMult", fontsize=20)
plt.ylabel("TPC refMult", fontsize=20)
# plt.colorbar()
'''
plt.show()
