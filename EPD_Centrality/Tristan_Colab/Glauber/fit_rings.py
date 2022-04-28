import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
from scipy.optimize import curve_fit
import fit_rings_functions as frf
from numpy.random import default_rng
rng = default_rng()

# Load the outer ring sum and n_coll/n_part data.
ring_sum, n_coll, n_part, predictions, refmult = frf.load_data()
size = int(len(n_part))
# Put ring sums in a histogram.
ring_max = int(max(ring_sum)*1.2)
bin_val = 1600
bin_range = (0, bin_val)
x_cutoff = 150
ring_hist = np.histogram(ring_sum, bins=bin_val,  range=bin_range, density=True)[0]
density = True

ring_sum = predictions[2]

# Just a thing to test out some known GMC parameters.
alpha = 0.78
p = 0.83195
n = 3.75
bins = 750
gmc = frf.gmc_dist_generator(n_coll, n_part, n, p, alpha)
np.save(r'C:\Users\dansk\Documents\Thesis\Protons\2022_urqmd\pro_count_archives\1M_events_refmult3_0.4_2'
        r'.0\gmc_arr.npy', gmc)
# Now to get the RefMult3 values from UrQMD
ref_count = np.load(r'C:\Users\dansk\Documents\Thesis\Protons\2022_urqmd\pro_count_archives\1M_events_refmult3_0.4_2'
                    r'.0\pro_counts.npy', allow_pickle=True).astype(float)
ref_bin = np.load(r'C:\Users\dansk\Documents\Thesis\Protons\2022_urqmd\pro_count_archives\1M_events_refmult3_0.4_2'
                  r'.0\pro_bins.npy', allow_pickle=True)
refmult = np.repeat(ref_bin[2][0][0][:-1], np.sum(ref_count[2][0], axis=1).astype('int'))
count, bins = np.histogram(refmult, bins=bins, range=(0, bins))
count = count/count[100]
count = np.hstack((0, count))
plt.plot(bins, count, label=r'RefMult3', color='red')
count, bins = np.histogram(gmc, bins=bins, range=(0, bins))
count = count/count[100]
count = np.hstack((0, count))
plt.plot(bins, count, label=r'GMC', color='blue', alpha=0.5)
# plt.hist(refmult, bins=bins, range=(0, bins), histtype='step', label=r'RefMult3', color='red')
# plt.hist(gmc, bins=bins, range=(0, bins), histtype='step', label=r"GMC", color='blue')
plt.legend()
plt.yscale('log')
plt.title(r"$\sqrt{s_{NN}}$=14.6 GeV UrQMD", fontsize=30)
plt.xlabel("Mult", fontsize=20)
plt.ylabel(r"$\frac{N}{N[300]}$", fontsize=20)
plt.tight_layout()
plt.show()

"""
Now to get a sense of how the GMC generated distributions depend on the parameters. We'll
make two plots where only alpha is varied, two where only n is varied, and two where only
p is varied. Bear in mind, alpha is in the range [0, 1] and p is in the range(0, 1]. n is
wide open (apart from 0), but once we get a feel for what it does we can set a realistic
range for our data set.
"""

gmc_alpha_high = frf.gmc_dist_generator(n_coll, n_part, 5, 0.7, 1)  # All n_coll
gmc_alpha_low = frf.gmc_dist_generator(n_coll, n_part, 5, 0.7, 0)  # All n_part
gmc_n_high = frf.gmc_dist_generator(n_coll, n_part, 10, 0.5, 0.5)
gmc_n_low = frf.gmc_dist_generator(n_coll, n_part, 5, 0.5, 0.5)
gmc_p_high = frf.gmc_dist_generator(n_coll, n_part, 5, 0.6, 0.5)
gmc_p_low = frf.gmc_dist_generator(n_coll, n_part, 5, 0.4, 0.5)
bins = 50
fig, ax = plt.subplots(1, 3, figsize=(16, 9), constrained_layout=True)
ax[0].hist(ring_sum, bins=bins, histtype='step', label=r'$\Sigma EPD_{outer}$', color='red', density=density)
ax[0].hist(gmc_alpha_high, bins=bins, histtype='step', label=r"$\alpha$ = 1", density=density)
ax[0].hist(gmc_alpha_low, bins=bins, histtype='step', label=r"$\alpha$ = 0", density=density)
ax[0].legend()
ax[0].set_yscale('log')
ax[1].hist(ring_sum, bins=bins, histtype='step', label=r'$\Sigma EPD_{outer}$', color='red', density=density)
ax[1].hist(gmc_n_high, bins=bins, histtype='step', label=r"n = 10", density=density)
ax[1].hist(gmc_n_low, bins=bins, histtype='step', label=r"n = 5", density=density)
ax[1].legend()
ax[1].set_yscale('log')
ax[2].hist(ring_sum, bins=bins, histtype='step', label=r'$\Sigma EPD_{outer}$', color='red', density=density)
ax[2].hist(gmc_p_high, bins=bins, histtype='step', label=r"p = 0.6", density=density)
ax[2].hist(gmc_p_low, bins=bins, histtype='step', label=r"p = 0.4", density=density)
ax[2].legend()
ax[2].set_yscale('log')
plt.show()

"""
We can see several things from this plot:
-Increasing alpha will increase the knee and the max, but not the MPV.
-Increasing n will increase both the knee and the MPV, but not the max.
-Increasing p will decrease the knee and the MPV, but not the max.

How these work in concert will be important, but we need a baseline to shoot for. In my
distribution (from 14.5 GeV collider data, "outside" EPD rings being ring# > 7), the
knee is from about 350 to 420 (the slopes look pretty steady before and after those points),
so I'll aim for that. All of the distributions made are too far out, so I'll have to
use lower alpha and n but higher p. This should have been clear for n as in pp collisions
the average produced particle number should be in the single digits at 14.5 GeV.

First, let's look at how far we can go with just n and just p with alpha at 0.5.
"""

gmc_n_high = frf.gmc_dist_generator(n_coll, n_part, 4, 0.5, 0.5)
gmc_n_low = frf.gmc_dist_generator(n_coll, n_part, 1, 0.5, 0.5)
gmc_p_high = frf.gmc_dist_generator(n_coll, n_part, 5, 0.9, 0.5)
gmc_p_low = frf.gmc_dist_generator(n_coll, n_part, 5, 0.8, 0.5)

fig, ax = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)
ax[0].hist(ring_sum, bins=bins, histtype='step', label=r'$\Sigma EPD_{outer}$', color='red', density=density)
ax[0].hist(gmc_n_high, bins=bins, histtype='step', label=r"n = 4", density=density)
ax[0].hist(gmc_n_low, bins=bins, histtype='step', label=r"n = 1", density=density)
ax[0].legend()
ax[0].set_yscale('log')
ax[1].hist(ring_sum, bins=bins, histtype='step', label=r'$\Sigma EPD_{outer}$', color='red', density=density)
ax[1].hist(gmc_p_high, bins=bins, histtype='step', label=r"p = 0.9", density=density)
ax[1].hist(gmc_p_low, bins=bins, histtype='step', label=r"p = 0.8", density=density)
ax[1].legend()
ax[1].set_yscale('log')
plt.show()

"""
Setting p higher seems to have closer to the desired effect, but we have a problem: the
distribution is not flat enough to match our ring_sum distribution. This is in part fine
as the low centrality area will not be well modeled by the GMC, but the mid range will
not fit here.

From our observations above, lowering alpha should help flatten the distribution at
lower n and higher p. So let's have the same plots but now with lower alpha. I'll use
the lower/higher values for n/p for comparison.
"""

gmc_n_high = frf.gmc_dist_generator(n_coll, n_part, 1, 0.5, 0.5)
gmc_n_low = frf.gmc_dist_generator(n_coll, n_part, 1, 0.5, 0.1)
gmc_p_high = frf.gmc_dist_generator(n_coll, n_part, 5, 0.9, 0.5)
gmc_p_low = frf.gmc_dist_generator(n_coll, n_part, 5, 0.9, 0.1)

fig, ax = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)
ax[0].hist(ring_sum, bins=bins, histtype='step', label=r'$\Sigma EPD_{outer}$', color='red', density=density)
ax[0].hist(gmc_n_high, bins=bins, histtype='step', label=r"$\alpha$ = 0.5", density=density)
ax[0].hist(gmc_n_low, bins=bins, histtype='step', label=r"$\alpha$ = 0.1", density=density)
ax[0].set_title("n=1, p=0.5", fontsize=25)
ax[0].legend()
ax[0].set_yscale('log')
ax[1].hist(ring_sum, bins=bins, histtype='step', label=r'$\Sigma EPD_{outer}$', color='red', density=density)
ax[1].hist(gmc_p_high, bins=bins, histtype='step', label=r"$\alpha$ = 0.5", density=density)
ax[1].hist(gmc_p_low, bins=bins, histtype='step', label=r"$\alpha$ = 0.1", density=density)
ax[1].set_title("n=5, p=0.9", fontsize=25)
ax[1].legend()
ax[1].set_yscale('log')
plt.show()

"""
It seems the best fit so far is at n=1, p=0.5, and alpha=0.4. But it needs to be a bit flatter, so
I'm going to use my prior observations and move n around until the knee is at the same spot, then 
do the same with apha and p. I'll also turn the normalisation off and try to scale the distributions 
to one another. In order to do this last part, I'm going to scale to the ring_sum distribution at
~200 in x.
"""

bins = 1800
scaler_rings, x = np.histogram(ring_sum, bins=bins, range=(0, bins))
x = x[:-1]
match = 363
gmc_n_high = frf.gmc_dist_generator(n_coll, n_part, 1.0, 0.5, 0.4)
gmc_n_mid = frf.gmc_dist_generator(n_coll, n_part, 0.8, 0.5, 0.4)
gmc_n_low = frf.gmc_dist_generator(n_coll, n_part, 0.7, 0.5, 0.4)
scaler_high_n = np.histogram(gmc_n_high, bins=bins, range=(0, bins))[0]
scaler_high_n = scaler_high_n * (scaler_rings[match]/scaler_high_n[match])
scaler_mid_n = np.histogram(gmc_n_mid, bins=bins, range=(0, bins))[0]
scaler_mid_n = scaler_mid_n * (scaler_rings[match]/scaler_mid_n[match])
scaler_low_n = np.histogram(gmc_n_low, bins=bins, range=(0, bins))[0]
scaler_low_n = scaler_low_n * (scaler_rings[match]/scaler_low_n[match])
gmc_p_high = frf.gmc_dist_generator(n_coll, n_part, 0.9, 0.5, 0.4)
gmc_p_mid = frf.gmc_dist_generator(n_coll, n_part, 0.9, 0.45, 0.4)
gmc_p_low = frf.gmc_dist_generator(n_coll, n_part, 0.9, 0.4, 0.4)
scaler_high_p = np.histogram(gmc_p_high, bins=bins, range=(0, bins))[0]
scaler_high_p = scaler_high_p * (scaler_rings[match]/scaler_high_p[match])
scaler_mid_p = np.histogram(gmc_p_mid, bins=bins, range=(0, bins))[0]
scaler_mid_p = scaler_high_p * (scaler_rings[match]/scaler_mid_p[match])
scaler_low_p = np.histogram(gmc_p_low, bins=bins, range=(0, bins))[0]
scaler_low_p = scaler_low_p * (scaler_rings[match]/scaler_low_p[match])
gmc_alpha_high = frf.gmc_dist_generator(n_coll, n_part, 0.9, 0.45, 0.4)
gmc_alpha_mid = frf.gmc_dist_generator(n_coll, n_part, 0.9, 0.45, 0.3)
gmc_alpha_low = frf.gmc_dist_generator(n_coll, n_part, 0.9, 0.45, 0.2)
scaler_high_a = np.histogram(gmc_alpha_high, bins=bins, range=(0, bins))[0]
scaler_high_a = scaler_high_a * (scaler_rings[match]/scaler_high_a[match])
scaler_mid_a = np.histogram(gmc_alpha_mid, bins=bins, range=(0, bins))[0]
scaler_mid_a = scaler_mid_a * (scaler_rings[match]/scaler_mid_a[match])
scaler_low_a = np.histogram(gmc_alpha_low, bins=bins, range=(0, bins))[0]
scaler_low_a = scaler_low_a * (scaler_rings[match]/scaler_low_a[match])
density = True
bins = 50
fig, ax = plt.subplots(1, 3, figsize=(16, 9), constrained_layout=True)
ax[0].plot(scaler_rings, label=r'$\Sigma EPD_{outer}$', color='red')
ax[0].plot(scaler_high_n, label=r"n = 1.0")
ax[0].plot(scaler_mid_n, label=r"n = 0.8")
ax[0].plot(scaler_low_n, label=r"n = 0.7")
ax[0].legend()
ax[0].set_title(r"$\alpha$ = 0.4, p = 0.5")
ax[0].set_yscale('log')
ax[1].plot(scaler_rings, label=r'$\Sigma EPD_{outer}$', color='red')
ax[1].plot(scaler_high_p, label=r"p = 0.5")
ax[1].plot(scaler_mid_p, label=r"p = 0.45")
ax[1].plot(scaler_low_p, label=r"p = 0.4")
ax[1].legend()
ax[1].set_yscale('log')
ax[1].set_title(r"$\alpha$ = 0.4, n = 0.9")
ax[2].plot(x[:600], scaler_rings[:600], label=r'$\Sigma EPD_{outer}$', color='red')
ax[2].plot(x[:600], scaler_high_a[:600], label=r"$\alpha$ = 0.4")
ax[2].plot(x[:600], scaler_mid_a[:600], label=r"$\alpha$ = 0.3")
ax[2].plot(x[:600], scaler_low_a[:600], label=r"$\alpha$ = 0.2")
ax[2].legend()
ax[2].set_yscale('log')
ax[2].set_title(r"n = 0.9, p = 0.45")
plt.show()

# TODO Make this, you know, actually work.
# Set up the GMC arrays/histograms to find fits.
alpha = np.linspace(0.1, 0.3, 100)
n = np.linspace(0.7, 1.2, 100)
p = np.linspace(0.6, 0.4, 100)
hist_fits = []
mins = np.zeros((len(n), len(p)))

"""
n_pp_arrs = []
for i in n:
    for j in p:
        for k in alpha:
            n_pp = frf.gmc_dist_generator(n_coll, n_part, i, j, k)
            n_pp_arrs.append(n_pp)
"""

for i in range(len(n)):
    print("n:", i+1, "of", len(n))
    hist_fits.append([])
    for j in range(len(p)):
        hist_fits[i].append([])
        for k in range(len(alpha)):
            n_pp = frf.gmc_dist_generator(n_coll, n_part, n[i], p[j], alpha[k])
            count = np.histogram(n_pp, bins=bin_val,  range=bin_range, density=True)[0]
            mse = np.sum(np.power(np.subtract(count[x_cutoff:ring_max], ring_hist[x_cutoff:ring_max]), 2))/size
            hist_fits[i][j].append(mse)
        mins[i][j] = np.min(hist_fits[i][j])
hist_fits = np.array(hist_fits)

# Here's a heat map for the MSE values for the given n and p inputs
# (note: this only shows for the lowest alpha).
X, Y = np.meshgrid(np.linspace(0, len(n)-1, len(n)), np.linspace(0, len(p)-1, len(p)))
plt.pcolormesh(X, Y, mins, cmap="jet", norm=colors.LogNorm(), shading="auto")
plt.xlabel("n parameter", fontsize=15)
plt.ylabel("p parameter", fontsize=15)
plt.title(r"MSE Minima for GMC Fits", fontsize=20)
plt.colorbar()
plt.show()
plt.close()

# Here's where we test how the fitted results are looking.
nbd = rng.negative_binomial(n[1], p[3], int(1e6))
n_pp = rng.choice(nbd, size)
hist_fits_t = []
for k in range(len(alpha)):
    gmc = np.multiply(n_pp, np.add(alpha[k] * n_coll, (1 - alpha[k]) * n_part))
    count = np.histogram(gmc, bins=bin_val, range=bin_range, density=True)[0]
    mse = np.sum(np.power(np.subtract(count[x_cutoff:], ring_hist[x_cutoff:]), 2)) / size
    hist_fits_t.append(mse)
minim = np.where(np.min(hist_fits_t))[0]
gmc = np.multiply(n_pp, np.add(alpha[minim] * n_coll, (1 - alpha[minim]) * n_part))
print(gmc)

plt.hist(ring_sum, bins=bin_val,  range=bin_range, density=True,
         histtype="step", label=r"$\Sigma_{EPDrings}$")
plt.hist(gmc, bins=bin_val,  range=bin_range, density=True,
         histtype="step", label="GMC")
plt.legend()
plt.yscale("log")
plt.show()
