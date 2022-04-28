# \brief UrQMD analysis of net proton cumulants using various
#        centrality determination methods.
#
#
# \author Skipper Kagamaster
# \date 09/23/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import functions as fn
import os
import pandas as pd
import uproot as up

energy = 7  # 200 GeV, 27.7 (27) GeV, and 19.6 (19) GeV are the current options.
gev = 7.7
if energy == 200:
    gev = 200
elif energy == 27:
    gev = 27.7
elif energy == 19:
    gev = 19.6
elif energy == 15:
    gev = 14.5

"""
First, we'll import the relevant data. All data should have gone through proper QA; here,
we're just dealing with UrQMD so not a lot of vetting is necessary. All collisions in this
analysis were done at z=0, and we restricted p_T tot he interval 0.4-2.0 (as is common on
net proton cumulant analyses in STAR).
"""
os.chdir(r"C:\Users\dansk\Documents\Thesis\2021_DNP_Work\ML_fits\UrQMD")
if energy > 15:
    df = pd.read_pickle(r"C:\Users\dansk\Documents\Thesis\2021_DNP_Work\urqmd_{}.pkl".format(energy))
    ring_sum = np.zeros(len(df))
    for i in range(16):
        ring_sum = np.add(ring_sum, df['ring{}'.format(i+1)].to_numpy())
    refmult = df['refmult'].to_numpy()  # Charged particles in |eta| < 1.0
    refmult3 = df['refmult3'].to_numpy()  # Charged particles less protons/antiprotons
    index = (refmult >= 0) & (refmult3 >= 0)  # To get rid of events selected out by QA
    ring_sum = ring_sum[index]
    refmult = refmult[index]
    refmult3 = refmult3[index]
    b = df['b'].to_numpy()[index]  # Impact parameter
    nprotons = df['net_protons'].to_numpy()[index]  # Net proton count in |eta| < 1.0
    # These are the ML predictions trained to match b.
    linear = np.load("{}linear_predictions.npy".format(energy), allow_pickle=True)  # [index]
    relu = np.load("{}relu_predictions.npy".format(energy), allow_pickle=True)  # [index]
    swish = np.load("{}swish_predictions.npy".format(energy), allow_pickle=True)  # [index]
    mish = np.load("{}mish_predictions.npy".format(energy), allow_pickle=True)  # [index]
    predictions = np.load(r"C:\Users\dansk\Documents\Thesis\2021_DNP_Work\predictions_{}.npy".format(energy),
                          allow_pickle=True)
    # Putting them all together.
    cent_arr = np.vstack((refmult, refmult3, ring_sum, linear, relu, swish, mish, b))
    cent_labels = ["RefMult", "RefMult3", r"$X_{FWD}$", r"$X_{\zeta',LW}$",
                   r"$X_{\zeta',relu}$", r"$X_{\zeta',swish}$",
                   r"$X_{\zeta',mish}$", "b"]
    cent_arr = np.vstack((refmult3, predictions, b))
    cent_labels = ["RefMult3", r"$X_{FWD}$", r"$X_{\zeta',LW}$",
                   r"$X_{\zeta',relu}$", r"$X_{\zeta',swish}$",
                   r"$X_{\zeta',mishCNN}$", "b"]

else:
    # Just for the 7.7 GeV stuff for DNP
    data = up.open(r"D:\UrQMD_cent_sim\{}\CentralityNtuple.root".format(energy))['Rings']
    glob = 'b'
    b = data['b'].array(library='np')
    refmult3 = data['RefMult3'].array(library='np')
    ring_sum = np.zeros(len(refmult3))
    for i in range(16):
        ring_sum += data['r%02d' % (i+1)].array(library='np')
    os.chdir(r"C:\Users\dansk\Documents\Thesis\2021_DNP_Work")
    # These are the ML predictions trained to match b.
    mish = np.load("predictions_{}_".format(energy)+glob+".npy", allow_pickle=True)[3]  # [index]
    linear = np.load("predictions_{}_".format(energy)+glob+".npy", allow_pickle=True)[0]  # [index]
    relu = np.load("predictions_{}_".format(energy)+glob+".npy", allow_pickle=True)[1]  # [index]
    swish = np.load("predictions_{}_".format(energy)+glob+".npy", allow_pickle=True)[2]
    # Putting them all together.
    cent_arr = np.vstack((refmult3, ring_sum, linear, swish, mish, b))
    cent_labels = ["RefMult3", r"$X_{\Sigma \zeta'}$", r"$X_{\zeta',LW}$",
                   r"$X_{\zeta',swish}$",
                   r"$X_{\zeta',mishCNN}$", "b"]
    """
    for refmult, 7.7:

 [  3.76978621  16.88260562  14.69025585   9.1671573    6.58485723
    5.24342389   4.50977399   4.53832742  10.6732547    2.54398835]
 [  2.61680376  12.70253989   9.4977327    6.4030593    4.78516659
    3.64718383   2.86564362   2.17586398   3.87865305   1.53769148]
 [  2.46483285  11.45453379   9.17743812   6.37953183   4.82650787
    3.65682101   2.86621113   2.16597148   3.88892198   1.54241568]
    for b:
     [  3.54745456  16.56965993  15.53399341   7.7280567    4.92848995
    3.71684525   3.10031343   2.6862325    6.64990945   1.97637224]
 [  2.43818414  11.20831083   9.14071619   6.42747585   4.75382179
    3.6298835    2.85896353   2.15875598   3.85174926   1.53218667]
 [  2.46471226  11.45547487   9.2067669    6.4442491    4.82502997
    3.71618178   2.93354629   2.19965206   3.93942127   1.54618411]
    """

# A simple plot of the distributions, normalised in x and y.
bins = 50
lw = 3
plt.figure(figsize=(16, 9), constrained_layout=True)
for i in range(len(cent_arr)):
    plt.hist(fn.x_normalise(cent_arr[i]), bins=bins, histtype='step', density=True,
             label=cent_labels[i], alpha=0.5, lw=lw)
plt.legend()
plt.yscale('log')
plt.title("Normalised Distributions for {} GeV".format(gev), fontsize=30)
plt.xlabel("Centrality (C, AU)", fontsize=20)
plt.ylabel(r"$\frac{dN}{dC}$", fontsize=20)
# plt.show()
plt.close()

"""
Now we will make cumulants based on quantile selection. Note that RefMult and RefMult3
are anti-correlated with b, so we will have quantiles in the opposite direction.
"""
cent_ranges = [20, 30, 40, 50, 60, 70, 80, 90, 95]
cent_ranges_b = [80, 70, 60, 50, 40, 30, 20, 10, 5]
centralities = np.zeros((len(cent_arr), len(cent_ranges)))
x_arr = []
x_arr_cbwc = []
phi = []
if glob == 'b':
    count = 2
else:
    count = 5
for i in range(len(centralities)):
    print(cent_labels[i])
    if i < count:
        centralities[i] = np.percentile(cent_arr[i], cent_ranges, interpolation='lower')
        phi.append(fn.b_var(cent_arr[i], cent_arr[-1], centralities[i]))
        if energy > 15:
            x_arr.append(fn.cbwc(cent_arr[i], nprotons, centralities[i]))
            x_arr_cbwc.append(fn.cbwc_dnp(cent_arr[i], nprotons, centralities[i]))
    else:
        centralities[i] = np.percentile(cent_arr[i], cent_ranges_b, interpolation='lower')
        phi.append(fn.b_var_b(cent_arr[i], cent_arr[-1], centralities[i]))
        if energy > 15:
            x_arr.append(fn.cbwc_b(cent_arr[i], nprotons, centralities[i]))
            x_arr_cbwc.append(fn.cbwc_dnp_b(cent_arr[i], nprotons, centralities[i]))
x_arr = np.asarray(x_arr)
phi = np.asarray(phi)
print(phi[:, 1])
print(phi[:, 1, 0])
for i in range(len(phi[0])):
    for j in range(len(phi[i][1])):
        arr = np.array((phi[i][0][j], phi[-1][0][j]))
        err_arr = np.array((phi[i][1][j], phi[-1][1][j]))
        q = phi[i][0][j]/phi[-1][0][j]
        phi[i][1][j] = fn.err_ratio(q, arr, err_arr)
    phi[i][0] = np.divide(phi[i][0], phi[-1][0])
xrange = np.arange(len(cent_ranges)+1)
labels = ["80-100%", "70-80%", "60-70%", "50-60%", "40-50%",
          "30-40%", "20-30%", "10-20%", "5-10%", "0-5%"]
marker = ["o", "*", "^", "v", ">", "<", "X", "P"]
lister = [0, 1, 2, 4]
plt.figure(figsize=(12, 5))
for i in lister:  # range(len(cent_arr)-1):
    plt.errorbar(x=labels, y=phi[i][0], yerr=phi[i][1], label=cent_labels[i], marker=marker[i], ms=8,
                 elinewidth=1, capsize=2)
plt.legend()
plt.xticks(xrange, labels=labels, rotation=45)
plt.xlabel("Centrality", fontsize=20)
plt.ylabel(r"$\Phi=\frac{\sigma^2_{x,i}}{\sigma^2_{b,i}}$", fontsize=20)
plt.title("Resolution for {} GeV".format(gev), fontsize=30)
plt.yscale('log')
plt.tight_layout()
plt.show()
"""
x = []
y = []
print("Determining cumulants for:")
for i in range(len(cent_arr)):
    print(cent_labels[i])
    if i < 2:
        first, second = fn.cumulants(cent_arr[i], nprotons, reverse=False)
    else:
        first, second = fn.cumulants(cent_arr[i], nprotons, reverse=True)
        first = first[::-1]
        for j in second:
            second[j] = second[j][::-1]
    x.append(first/np.max(first))
    y.append(second)
# x, y = fn.cumulants(refmult3, nprotons)
fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
for i in range(2):
    for j in range(2):
        r = i*2 + j
        for k in range(len(cent_arr)):
            ax[i, j].plot(x[k], y[k][r], label=cent_labels[k])
ax[0, 0].legend()
plt.show()
plt.close()
"""
fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
cumulant = [r"$C_1$", r"$C_2$", r"$C_3$", r"$C_4$"]
for i in range(2):
    for j in range(2):
        r = i*2 + j
        for k in range(len(cent_arr)):
            # ax[i, j].errorbar(xrange+(0.05*k), x_arr[k][r][0], yerr=x_arr[k][r][1], lw=0,
            #                  markersize=6, elinewidth=1, capsize=2, label=cent_labels[k], alpha=0.5,
            #                  marker=marker[k])
            ax[i, j].plot(xrange+(0.05*k), x_arr_cbwc[k][r], lw=0, markersize=6,
                          label=cent_labels[k], alpha=0.5, marker=marker[k])
        ax[i, j].set_xlabel("Centrality", fontsize=15)
        ax[i, j].set_ylabel(cumulant[r], fontsize=15)
        ax[i, j].set_xticks(xrange)
        ax[i, j].set_xticklabels(labels, rotation=45)
ax[0, 0].legend()
fig.suptitle(r"UrQMD $\sqrt{s_{NN}}$" + " = {} GeV".format(gev), fontsize=20)
plt.show()

for i in range(len(cent_arr)):
    x_arr_cbwc[i][2] = np.divide(x_arr_cbwc[i][2], x_arr_cbwc[i][1])
    x_arr_cbwc[i][3] = np.divide(x_arr_cbwc[i][3], x_arr_cbwc[i][1])
    x_arr_cbwc[i][1] = np.divide(x_arr_cbwc[i][1], x_arr_cbwc[i][0])
fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
cumulant = [r"$\mu$", r"$\frac{\sigma^2}{\mu}$", r"$S\sigma$", r"$\kappa\sigma^2$"]
for i in range(2):
    for j in range(2):
        r = i*2 + j
        for k in range(2):  # range(len(cent_arr)):
            # ax[i, j].errorbar(xrange+(0.05*k), x_arr[k][r][0], yerr=x_arr[k][r][1], lw=0,
            #                  markersize=6, elinewidth=1, capsize=2, label=cent_labels[k], alpha=0.5,
            #                  marker=marker[k])
            ax[i, j].plot(xrange+(0.05*k), x_arr_cbwc[k][r], lw=0, markersize=6,
                          label=cent_labels[k], alpha=0.5, marker=marker[k])
        ax[i, j].set_xlabel("Centrality", fontsize=15)
        ax[i, j].set_ylabel(cumulant[r], fontsize=15)
        ax[i, j].set_xticks(xrange)
        ax[i, j].set_xticklabels(labels, rotation=45)
fig.suptitle(r"UrQMD $\sqrt{s_{NN}}$" + " = {} GeV".format(gev), fontsize=20)
ax[0, 0].legend()
plt.show()
