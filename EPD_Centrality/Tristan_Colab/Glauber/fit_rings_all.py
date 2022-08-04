import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
from scipy.optimize import curve_fit
import fit_rings_functions as frf
from matplotlib.colors import LogNorm
from numpy.random import default_rng
rng = default_rng()

# Load the refmult, ring, and n_coll/n_part data.
n_coll, n_part, predictions = frf.load_data()
density = True

# I'm going to plot the best fits, so far, as distributions.
dists = [r"$X_{RM3}$", r"$X_{\Sigma}$", r"$X_{\Sigma_{out}}$",
         r"$X_{LW}$", r"$X_{ReLU}$"]
cent_range = (95, 90, 80, 70, 60, 50, 40, 30, 20, 10)
GMC = []
gmc = frf.gmc_dist_generator(n_coll, n_part, 2.0, 0.7, 0.5)
print(np.percentile(gmc, cent_range)[::-1])
GMC.append(gmc)
gmc = frf.gmc_dist_generator(n_coll, n_part, 2.0, 0.7, 0.5)
GMC.append(gmc)
gmc = frf.gmc_dist_generator(n_coll, n_part, 2.0, 0.8, 1.0)
GMC.append(gmc)
gmc = frf.gmc_dist_generator(n_coll, n_part, 3.0, 0.8, 0.5)
GMC.append(gmc)
gmc = frf.gmc_dist_generator(n_coll, n_part, 1.85, 0.65, 0.28)
print(np.percentile(gmc, cent_range)[::-1])
GMC.append(gmc)

bins = 850

fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
big_set = (0, 1, 3, 4)
for i in range(2):
    for j in range(2):
        x = big_set[i*2 + j]
        count, bins = np.histogram(predictions[x], bins=bins, range=(-100, bins-100))
        gmc_count, gmc_bins = np.histogram(GMC[x], bins=bins, range=(-100, bins-100))
        gmc_count = gmc_count*(count[300]/gmc_count[300])
        ax[i, j].plot(bins[:-1], count, label=dists[x], color='red', lw=2)
        ax[i, j].plot(bins[:-1], gmc_count, label="GMC", color='b', lw=2, alpha=0.6)
        ax[i, j].legend()
        ax[i, j].set_xlabel("X (AU)", fontsize=15)
        ax[i, j].set_ylabel(r"N (normalised)", fontsize=15)
        ax[i, j].set_yscale('log')
plt.show()
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
small_set = (0, -1)
for i in range(2):
    x = small_set[i]
    count, bins = np.histogram(predictions[x], bins=bins, range=(-100, bins - 100))
    gmc_count, gmc_bins = np.histogram(GMC[x], bins=bins, range=(-100, bins - 100))
    gmc_count = gmc_count * (count[300] / gmc_count[300])
    ax[i].plot(bins[:-1], count, label=dists[x], color='red', lw=2)
    ax[i].plot(bins[:-1], gmc_count, label="GMC", color='b', lw=2, alpha=0.6)
    ax[i].legend()
    ax[i].set_xlabel("X (AU)", fontsize=15)
    ax[i].set_ylabel(r"N (normalised)", fontsize=15)
    ax[i].set_yscale('log')
plt.show()
plt.close()

chi_2 = np.zeros((2, 10, 10, 10))
n = np.linspace(1.3, 2, 10)
p = np.linspace(0.7, 0.9, 10)
alpha = np.linspace(0.1, 0.3, 10)
print("Working on n =")
for i in range(10):
    print(n[i])
    for j in range(10):
        for k in range(10):
            gmc = frf.gmc_dist_generator(n_coll, n_part, n[i], p[j], alpha[k])
            count = np.histogram(gmc, bins=700, range=(0, 700), density=True)[0][100:]
            count_r = np.histogram(predictions[0], bins=700, range=(0, 700), density=True)[0][100:]
            count_re = np.histogram(predictions[4], bins=700, range=(0, 700), density=True)[0][100:]
            chi_2[0][i][j][k] = np.corrcoef(count, count_r)[0][1]
            chi_2[1][i][j][k] = np.corrcoef(count, count_re)[0][1]
print("Correlations found.")
np.save(r'D:\14GeV\Thesis\Proton_Analysis_WIP\ML\chi_2.npy', chi_2)
fig, ax = plt.subplots(3, 4, constrained_layout=True, figsize=(16, 9))
for i in range(3):
    for j in range(4):
        x = i*4 + j
        if x >= 10:
            ax[i, j].set_axis_off()
            continue
        X, Y = np.meshgrid(p, alpha)
        im = ax[i, j].pcolormesh(X, Y, chi_2[0][x].T, cmap='jet', norm=LogNorm())
        ax[i, j].set_title('n=' + str(n[x]), fontsize=15)
        ax[i, j].set_ylabel(r"$\alpha$", fontsize=15)
        ax[i, j].set_xlabel('p', fontsize=15)
        fig.colorbar(im, ax=ax[i, j])
plt.show()

"""
Let's first plot the various distributions on a normalised graph.
"""
# ***TURNED OFF TO SAVE TIME FOR NOW***
"""
plt.figure(figsize=(16, 9))
for i in data_df.columns:
    plt.hist(data_df[i].to_numpy(), bins=900, range=(-100, 800), histtype='step', label=i, alpha=0.7)
plt.yscale('log')
plt.legend()
plt.xlabel("C (AU)", fontsize=20)
plt.ylabel(r"$\frac{dN}{dC}$", fontsize=20)
plt.title("Reference Distributions", fontsize=30)
plt.tight_layout()
plt.show()
"""
# ***TURNED OFF TO SAVE TIME FOR NOW***
"""
Now to get a sense of how the GMC generated distributions depend on the parameters. We'll
make plots where only alpha is varied, only n is varied, and only p is varied. Bear in
mind, alpha is in the range [0, 1] and p is in the range(0, 1]. n is wide open (apart from 0),
but once we get a feel for what it does we can set a realistic range for our data set.
"""
# ***TURNED OFF TO SAVE TIME FOR NOW***
"""
gmc_alpha_high = frf.gmc_dist_generator(n_coll, n_part, 2, 0.8, 1)  # All n_coll
gmc_alpha_low = frf.gmc_dist_generator(n_coll, n_part, 2, 0.8, 0)  # All n_part
gmc_n_high = frf.gmc_dist_generator(n_coll, n_part, 3, 0.8, 0.5)
gmc_n_low = frf.gmc_dist_generator(n_coll, n_part, 1, 0.8, 0.5)
gmc_p_high = frf.gmc_dist_generator(n_coll, n_part, 2, 0.9, 0.5)
gmc_p_low = frf.gmc_dist_generator(n_coll, n_part, 2, 0.7, 0.5)
bins = 50
fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
for i in range(2):
    for j in range(3):
        x = i*3 + j
        epd = data_df[data_df.columns[x+1]].to_numpy()
        ax[i, j].hist(epd, bins=bins, histtype='step', label=data_df.columns[x+1], color='red', density=density)
        ax[i, j].hist(gmc_alpha_high, bins=bins, histtype='step', label=r"$\alpha$ high", density=density, alpha=0.6)
        ax[i, j].hist(gmc_alpha_low, bins=bins, histtype='step', label=r"$\alpha$ low", density=density, alpha=0.6)
        ax[i, j].hist(gmc_n_high, bins=bins, histtype='step', label=r"n high", density=density, alpha=0.6)
        ax[i, j].hist(gmc_n_low, bins=bins, histtype='step', label=r"n low", density=density, alpha=0.6)
        ax[i, j].hist(gmc_p_high, bins=bins, histtype='step', label=r"p high", density=density, alpha=0.6)
        ax[i, j].hist(gmc_p_low, bins=bins, histtype='step', label=r"p low", density=density, alpha=0.6)
        ax[i, j].set_yscale('log')
        ax[i, j].legend()
plt.show()
"""
# ***TURNED OFF TO SAVE TIME FOR NOW***
"""
Best ranges from these plots so far (n, p, alpha):
RefMult3: 2, 0.7, 0.5
ring_sum: 2, 0.7, 0.5
ring_sum_outer: 2, 0.8, 1
pred_linear: 3, 0.8, 0.5
pred_relu: 2, 0.7, 0.5
pred_swish: 2, 0.7, 0.5
pred_mish: 2, 0.7, 0.5

Some of these reflect the defaults I put in, like alpha=0.5. I need to have a more
complex relationship found for these, but this is a starting point.

Next, I should find some Pearson coefficients for these just to give a more numerical
reference. I'll keep it as above and see what it looks like.
"""
# ***TURNED OFF TO SAVE TIME FOR NOW***
"""
bins = 400
limits = (-300, 800)
pearson = np.empty((7, 6))
arr_a_high = np.histogram(gmc_alpha_high, bins=bins, range=limits)[0]
arr_a_low = np.histogram(gmc_alpha_low, bins=bins, range=limits)[0]
arr_n_high = np.histogram(gmc_n_high, bins=bins, range=limits)[0]
arr_n_low = np.histogram(gmc_n_low, bins=bins, range=limits)[0]
arr_p_high = np.histogram(gmc_p_high, bins=bins, range=limits)[0]
arr_p_low = np.histogram(gmc_p_low, bins=bins, range=limits)[0]
arr_gmc = np.array((arr_a_high, arr_a_low,
                    arr_n_high, arr_n_low,
                    arr_p_high, arr_p_low))
for i in range(len(data_df.columns)):
    x = data_df.columns[i]
    arr = np.histogram(data_df[x].to_numpy(), bins=bins, range=limits)[0]
    for j in range(6):
        pearson[i][j] = np.corrcoef(arr, arr_gmc[j])[0][1]
"""
# ***TURNED OFF TO SAVE TIME FOR NOW***
"""
And here they are:
[[0.82221065 0.75941789 0.68956273 0.9206971  0.92794668 0.65114815]
 [0.4987955  0.32567222 0.58557133 0.22993804 0.20767551 0.62953077]
 [0.85445445 0.74244348 0.89189297 0.62616461 0.59272886 0.88957073]
 [0.8455439  0.78543834 0.77789435 0.83013154 0.82212128 0.7478491 ]
 [0.8024473  0.74247924 0.66699162 0.91066982 0.91978587 0.62800276]
 [0.82307724 0.76403807 0.69108043 0.9236603  0.93089936 0.65334863]
 [0.82337566 0.76125208 0.68391491 0.92406811 0.93072462 0.64259499]]

Well. It's time to set some parameters and do a metric crapton of fits. Ranges should probably be:
2.0 < n < 3.5
0.6 < p < 0.9
0.4 < alpha < 1.0

If I do 10 bins for each, that's 1000 bins. So with 7 to fit, that's 7k histograms being
generated. Not the worst for stats (with 1M points, though, we're getting a little dicey).
Now to just find a way to visualise the metric. I can make a heat map easy enough, but
there are 4 variables (including the Pearson coefficient). So I think the best thing to 
do is a plot of 2 parameters with the Pearson value when using the third variable in the 
heat map, then 10 plots of that 3rd parameter. So this will net us 70 plots, which isn't
too arduous to look over. Then we can take the best range within those sets and reapply
until we reach some sort of convergence.

First, let's generate arrays of the histogram data.
hist_arr is an array for each of the 7 references which will have Pearson coefficients
as follows:
10 n arrays each with 10 p arrays each with 10 Pearson values for each alpha value.
So we will have 70, 10x10 matrices for our colormap plots. Values will be:
n values:
[2.0, 2.16666667, 2.33333333, 2.5, 2.66666667, 2.83333333, 3.0, 3.16666667, 3.33333333, 3.5]
p values:
[0.6, 0.63333333, 0.66666667, 0.7, 0.73333333, 0.76666667, 0.8, 0.83333333, 0.86666667, 0.9]
alpha values:
[0.4, 0.46666667, 0.53333333, 0.6, 0.66666667, 0.73333333, 0.8, 0.86666667, 0.93333333, 1.0]
"""
bins = 400
limits = (-300, 800)
hist_arr = np.zeros((7, 10, 10, 10))
n_vals = np.linspace(2, 3.5, 10)
p_vals = np.linspace(0.6, 0.9, 10)
a_vals = np.linspace(0.4, 1.0, 10)
r_vals = np.zeros((7, bins))
for i in range(len(data_df.columns)):
    x = data_df.columns[i]
    r_vals[i] = np.histogram(data_df[x].to_numpy(), bins=bins, range=limits)[0]
print("Starting GMC fit analysis.", "\n", "Working on:")
count = 0
for j in range(10):
    for k in range(10):
        x = j*10 + k
        print(x+1, "of 100.")
        count += 1
        nbd = frf.nbd_generator(n_vals[j], p_vals[k])
        for m in range(10):
            gmc = frf.gmc_short(nbd, a_vals[m], n_coll, n_part)
            gmc = np.histogram(gmc, bins=bins, range=limits)[0]
            for i in range(7):
                hist_arr[i][j][k][m] = np.corrcoef(r_vals[i], gmc)[0][1]
pred_loc = r"C:\Users\dansk\Documents\Thesis\2021_DNP_Work\ML_fits\9_2021\\"
np.save(pred_loc+"pearsons.npy", hist_arr)

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
It seems the best fit so far is at n=1, p=0.5, and alpha=0.1. But it needs to be a bit flatter, so
I'm going to use my prior observations and move n around until the knee is at the same spot, then 
do the same with apha and p. I'll also turn the normalisation off and try to scale the distributions 
to one another. In order to do this last part, I'm going to scale to the ring_sum distribution at
~200 in x.
"""
print("Generating results:")
bins = 1800
scaler_rings, x = np.histogram(ring_sum, bins=bins, range=(0, bins))
x = x[:-1]
match = 363
gmc_n_high = frf.gmc_dist_generator(n_coll, n_part, 3, 0.5, 0.1)
gmc_n_mid = frf.gmc_dist_generator(n_coll, n_part, 2, 0.5, 0.1)
gmc_n_low = frf.gmc_dist_generator(n_coll, n_part, 1, 0.5, 0.1)
scaler_high_n = np.histogram(gmc_n_high, bins=bins, range=(0, bins))[0]
scaler_high_n = scaler_high_n * (scaler_rings[match]/scaler_high_n[match])
scaler_mid_n = np.histogram(gmc_n_mid, bins=bins, range=(0, bins))[0]
scaler_mid_n = scaler_mid_n * (scaler_rings[match]/scaler_mid_n[match])
scaler_low_n = np.histogram(gmc_n_low, bins=bins, range=(0, bins))[0]
scaler_low_n = scaler_low_n * (scaler_rings[match]/scaler_low_n[match])
print("n list generated")
gmc_p_high = frf.gmc_dist_generator(n_coll, n_part, 1, 0.7, 0.1)
gmc_p_mid = frf.gmc_dist_generator(n_coll, n_part, 1, 0.5, 0.1)
gmc_p_low = frf.gmc_dist_generator(n_coll, n_part, 1, 0.3, 0.1)
scaler_high_p = np.histogram(gmc_p_high, bins=bins, range=(0, bins))[0]
scaler_high_p = scaler_high_p * (scaler_rings[match]/scaler_high_p[match])
scaler_mid_p = np.histogram(gmc_p_mid, bins=bins, range=(0, bins))[0]
scaler_mid_p = scaler_high_p * (scaler_rings[match]/scaler_mid_p[match])
scaler_low_p = np.histogram(gmc_p_low, bins=bins, range=(0, bins))[0]
scaler_low_p = scaler_low_p * (scaler_rings[match]/scaler_low_p[match])
print("p list generated")
gmc_alpha_high = frf.gmc_dist_generator(n_coll, n_part, 1.2, 0.53, 0)
gmc_alpha_mid = frf.gmc_dist_generator(n_coll, n_part, 1.2, 0.53, 0)
gmc_alpha_low = frf.gmc_dist_generator(n_coll, n_part, 1.2, 0.53, 0)
scaler_high_a = np.histogram(gmc_alpha_high, bins=bins, range=(0, bins))[0]
scaler_high_a = scaler_high_a * (scaler_rings[match]/scaler_high_a[match])
scaler_mid_a = np.histogram(gmc_alpha_mid, bins=bins, range=(0, bins))[0]
scaler_mid_a = scaler_mid_a * (scaler_rings[match]/scaler_mid_a[match])
scaler_low_a = np.histogram(gmc_alpha_low, bins=bins, range=(0, bins))[0]
scaler_low_a = scaler_low_a * (scaler_rings[match]/scaler_low_a[match])
plt.plot(x[:600], scaler_rings[:600], label=r'$\Sigma EPD_{outer}$', color='red')
plt.plot(x[:600], scaler_high_a[:600], label=r"n=1.2, p=0.53, $\alpha$=0", alpha=0.5)
plt.legend()
plt.title(r"GMC vs $\Sigma EPD_{outer}$", fontsize=30)
plt.xlabel("Reference (AU)", fontsize=20)
plt.ylabel("N", fontsize=20)
plt.yscale('log')
plt.show()
print(r"$\alpha$ list generated")
density = True
bins = 50
fig, ax = plt.subplots(1, 3, figsize=(16, 9), constrained_layout=True)
ax[0].plot(scaler_rings, label=r'$\Sigma EPD_{outer}$', color='red')
ax[0].plot(scaler_high_n, label=r"n = 3")
ax[0].plot(scaler_mid_n, label=r"n = 3")
ax[0].plot(scaler_low_n, label=r"n = 1")
ax[0].legend()
ax[0].set_title(r"$\alpha$ = 0.1, p = 0.5")
ax[0].set_yscale('log')
ax[1].plot(scaler_rings, label=r'$\Sigma EPD_{outer}$', color='red')
ax[1].plot(scaler_high_p, label=r"p = 0.7")
ax[1].plot(scaler_mid_p, label=r"p = 0.5")
ax[1].plot(scaler_low_p, label=r"p = 0.3")
ax[1].legend()
ax[1].set_yscale('log')
ax[1].set_title(r"$\alpha$ = 0.1, n = 1")
ax[2].plot(x[:600], scaler_rings[:600], label=r'$\Sigma EPD_{outer}$', color='red')
ax[2].plot(x[:600], scaler_high_a[:600], label=r"n=1.2, p=0.53", alpha=0.5)
ax[2].plot(x[:600], scaler_mid_a[:600], label=r"n=1.2, p=0.535", alpha=0.5)
ax[2].plot(x[:600], scaler_low_a[:600], label=r"n=1.2, p=0.54", alpha=0.5)
ax[2].legend()
ax[2].set_yscale('log')
ax[2].set_title(r"$\alpha$ = 0")
plt.show()

# Set up the GMC arrays/histograms to find fits.
alpha = np.linspace(0, 1, 4)
n = np.linspace(9.7, 10, 4)
p = np.linspace(0.12, 0.14, 4)
# hist_fits = []
# mins = np.zeros((len(n), len(p)))

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
        print("   p:", j + 1, "of", len(p))
        hist_fits[i].append([])
        nbd = rng.negative_binomial(n[i], p[j], int(1e6))
        for k in range(len(alpha)):
            gmc = np.add(alpha[k]*n_coll, (1-alpha[k])*n_part).astype("int")
            n_pp = []
            for l in gmc:
                n_pp.append(np.sum(rng.choice(nbd, l)))
            n_pp = np.array(n_pp)
            count = np.histogram(n_pp, bins=bin_val,  range=bin_range, density=True)[0]
            mse = np.sum(np.power(np.subtract(count[x_cutoff:ring_max], ring_hist[x_cutoff:ring_max]), 2))/size
            print("      alpha:", k + 1, "of", len(alpha), ", mse:", mse)
            hist_fits[i][j].append(mse)
        mins[i][j] = np.min(hist_fits[i][j])
hist_fits = np.array(hist_fits)
"""
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
