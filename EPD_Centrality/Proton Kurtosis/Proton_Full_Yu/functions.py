# \author Skipper Kagamaster
# \date 09/01/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
"""
These are the functions for proton cumulant QA and analysis.
"""

import uproot as up
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cbook
from scipy.stats import skew, kurtosis
import os
from pathlib import Path
from numpy.random import default_rng

rng = default_rng()


def gmc_dist_generator(n_coll, n_part, n, p, alpha):
    nbd = rng.negative_binomial(n, p, int(1e6))
    gmc = np.add(alpha * n_coll, (1 - alpha) * n_part).astype("int")
    n_pp = np.empty(len(gmc))
    for i in range(len(gmc)):
        n_pp[i] = np.sum(rng.choice(nbd, gmc[i]))
    return n_pp


def protons_my_data(loc=r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results\BigList",
                    low_bound=800, high_bound='max',
                    low_len=2000, high_len=6000, set_len=True,
                    std_mag=2, run_plot=True):
    """
    This function takes in data from my *.txt files generated from the pico reader and returns
    information from a histogram of refmult vs net protons. Why don't I just get the cumulants
    from the data? Good point. Shoot.
        :param loc: Where the files are located.
        :param low_bound: lower bound for the files (if truncating)
        :param high_bound: upper bound for the files; set to 'max' if no upper bound
        :param low_len: minimum number of events in a run
        :param high_len: maximum number of events in a run
        :param set_len: True if you want to truncate by event #, False otherwise
        :param std_mag: std deviations to cut <proton> and <antiproton> runs
        :param run_plot: True to plot the runs and cuts, False otherwise
    return:
        runs: list of all runs after QA
    return:
        pro_int_ave: net proton average per run, for graphing
    return:
        protons: array of refmult, protons, antiprotons, and net protons
    """
    os.chdir(loc)
    files = []
    for file in os.listdir():
        if file.startswith("out_20") & file.endswith(".txt"):
            files.append(file)

    runs = []
    protons = [[], [], [], []]
    events = []
    if high_bound == 'max':
        high_bound = len(files)
    files = files[low_bound:high_bound]
    pro_int_ave = []
    averages = [[], []]
    errs = [[], []]
    count = 0
    for file in files:
        data = np.loadtxt(file)
        length = len(data[:, 1])
        if set_len is True:
            if (length < low_len) | (length > high_len):
                continue
        runs.append(data[0][0])
        pro_int_ave.append([])
        events.append(length)
        for i in range(700):
            index = data[:, 1] == i
            if len(index) > 0:
                pro_int_ave[count].append(np.mean(data[:, 3][index]))
        protons[0].append(data[:, 1])
        protons[1].append(data[:, 2])
        protons[2].append(data[:, 3])
        protons[3].append(data[:, 4])
        averages[0].append(np.mean(data[:, 2]))
        averages[1].append(np.mean(data[:, 3]))
        errs[0].append(np.std(data[:, 2]) / np.sqrt(length))
        errs[1].append(np.std(data[:, 3]) / np.sqrt(length))
        count += 1

    runs = np.array(runs)
    pro_int_ave = np.array(pro_int_ave)
    pro_int_ave = np.where(pro_int_ave == np.nan, 0, pro_int_ave)
    events = np.array(events)
    protons = ak.Array(protons)
    averages = np.array(averages)
    errs = np.array(errs)
    ave = np.mean(averages, axis=1)
    std = np.std(averages, axis=1) * std_mag
    upper = averages + errs
    lower = averages - errs
    index_up0 = upper[0] >= ave[0] + std[0]
    index_up1 = upper[1] >= ave[1] + std[1]
    index_down0 = lower[0] <= ave[0] - std[0]
    index_down1 = lower[1] <= ave[1] - std[1]
    index_up = [index_up0, index_up1]
    index_down = [index_down0, index_down1]

    if run_plot is True:
        fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(16, 9))
        titles = [r'<Proton>', r'<Antiproton>']
        for i in range(2):
            x = np.linspace(0, len(averages[i]) - 1, len(averages[i]))
            ax[i].errorbar(x, averages[i], yerr=errs[i], capsize=3,
                           lw=0, marker='.', ms=5, elinewidth=1, c='black')
            ax[i].errorbar(x[index_up[i]],
                           averages[i][index_up[i]], yerr=errs[i][index_up[i]], capsize=3,
                           lw=0, marker='.', ms=5, elinewidth=1, c='r')
            ax[i].errorbar(x[index_down[i]],
                           averages[i][index_down[i]], yerr=errs[i][index_down[i]], capsize=3,
                           lw=0, marker='.', ms=5, elinewidth=1, c='r')
            ax[i].axhline(ave[i], c='green', label='mean')
            ax[i].axhline(ave[i] + std[i], c='blue', label=r'$3\sigma$')
            ax[i].axhline(ave[i] - std[i], c='blue')
            ax[i].legend()
            ax[i].set_xlabel("Run ID", fontsize=20)
            ax[i].set_ylabel(titles[i], fontsize=20)
        plt.show()

    upper0 = runs[index_up[0]]
    upper1 = runs[index_up[1]]
    lower0 = runs[index_down[0]]
    lower1 = runs[index_down[1]]
    bad_pros_up = np.unique(np.hstack((upper0, upper1)))
    bad_pros_down = np.unique(np.hstack((lower0, lower1)))
    bad_pros = np.unique(np.hstack((bad_pros_up, bad_pros_down)))

    good_runs = []
    for i in range(len(runs)):
        if runs[i] in bad_pros:
            continue
        else:
            good_runs.append(i)
    pros = [[], [], [], []]
    for i in range(4):
        pros[i].append(ak.flatten(protons[i][good_runs]))
    pros = ak.Array(pros)
    # Now to make a histogram which we can use to get our cumulants from.
    counter, binsX, binsY = np.histogram2d(ak.to_numpy(ak.flatten(protons[0][good_runs])),
                                           ak.to_numpy(ak.flatten(protons[3][good_runs])),
                                           bins=(700, 50),
                                           range=((0, 700), (0, 50)))
    refmult = ak.to_numpy(ak.flatten(protons[0]))
    net_protons = ak.to_numpy(ak.flatten(protons[3]))
    return runs[good_runs], pro_int_ave[good_runs], pros, refmult, net_protons


def protons_yu_data(file=r'C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results\moments.root',
                    data_cut=4):
    data = up.open(file)['PtTree']
    refmult = data['refMult3'].array(library='np')
    pro = data['Np'].array(library='np')
    apro = data['Nap'].array(library='np')
    protons = pro - apro
    r = int(len(protons) / data_cut)
    refmult = refmult[:r]
    protons = protons[:r]

    return refmult, protons


def protons_my_pandas(file=r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results\BigList\out_all.pkl",
                      data_cut=4):
    ring_set = np.array((7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32))
    data = pd.read_pickle(file)
    refmult = data["refmult"].to_numpy()
    net_protons = data["net_protons"].to_numpy()
    ring_sum = []
    for i in ring_set:
        ring_sum.append(data["ring{}".format(i)].to_numpy())
    ring_sum = np.array(ring_sum)
    ring_sum = np.sum(ring_sum, axis=0)
    r = int(len(refmult) / data_cut)
    refmult = refmult[:r]
    ring_sum = ring_sum[:r]
    net_protons = net_protons[:r]

    return refmult, ring_sum, net_protons


def protons_ml_data(loc=r"C:\Users\dansk\Documents\Thesis\2021_DNP_Work\ML_fits\9_2021\\"):
    refmult = np.load(loc+'refmult3.npy', allow_pickle=True)
    ring_sum = np.load(loc+'ring_sum.npy', allow_pickle=True)
    ring_sum_outer = np.load(loc+'ring_sum_outer.npy', allow_pickle=True)
    pred_l = np.load(loc+'linearpredictions.npy', allow_pickle=True)
    pred_r = np.load(loc+'relupredictions.npy', allow_pickle=True)
    pred_s = np.load(loc+'swishpredictions.npy', allow_pickle=True)
    pred_m = np.load(loc+'mishpredictions.npy', allow_pickle=True)

    predictions = np.asarray((ring_sum, ring_sum_outer, pred_l,
                              pred_r, pred_s, pred_m))

    return refmult, predictions


def plot_pro_ave(runs, pro_int_ave):
    """
    This will plot the average number of net protons per run.
    Params:
        runs: list of runs
        refmult: reference multiplicity
        protons: net proton array corresponding to refmult
    """
    r = len(pro_int_ave)
    plt.rcParams["figure.figsize"] = (7, 7)
    X, Y = np.meshgrid(np.linspace(0, len(runs) - 1, len(runs)), np.linspace(0, 699, 700))
    plt.pcolormesh(Y, X, pro_int_ave.T, norm=colors.LogNorm(), cmap="jet", shading="auto")
    plt.title(r"<Net Proton> vs RefMult, by Run", fontsize=25)
    plt.xlabel("RefMult3", fontsize=20)
    plt.ylabel("Run ID", fontsize=20)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def plot_pro_dist_refmult(protons):
    """
    This will plot the distributions of protons and antiprotons vs RefMult3.
    Params:
        protons: the refmult, proton, antiproton, and net proton array
    """
    ref_max = int(np.max(ak.to_numpy(ak.flatten(protons[0]))) * 1.1)
    pro_max = int(np.max(ak.to_numpy(ak.flatten(protons[1]))) * 1.3)
    apro_max = int(np.max(ak.to_numpy(ak.flatten(protons[2]))) * 1.5)
    fig, ax = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)
    count1, binsX1, binsY1 = np.histogram2d(ak.to_numpy(ak.flatten(protons[0])),
                                            ak.to_numpy(ak.flatten(protons[1])),
                                            bins=(ref_max, pro_max),
                                            range=((0, ref_max), (0, pro_max)))
    X, Y = np.meshgrid(binsX1, binsY1)
    im0 = ax[0].pcolormesh(X, Y, count1.T, norm=colors.LogNorm(), cmap='jet', shading='auto')
    ax[0].set_xlabel("RefMult3", fontsize=25)
    ax[0].set_ylabel("Protons", fontsize=25)
    fig.colorbar(im0, ax=ax[0])
    count2, binsX2, binsY2 = np.histogram2d(ak.to_numpy(ak.flatten(protons[0])),
                                            ak.to_numpy(ak.flatten(protons[2])),
                                            bins=(ref_max, apro_max),
                                            range=((0, ref_max), (0, apro_max)))
    X, Y = np.meshgrid(binsX2, binsY2)
    im1 = ax[1].pcolormesh(X, Y, count2.T, norm=colors.LogNorm(), cmap='jet', shading='auto')
    ax[1].set_xlabel("RefMult3", fontsize=25)
    ax[1].set_ylabel("Antiprotons", fontsize=25)
    fig.colorbar(im1, ax=ax[1])
    plt.show()
    plt.close()


# This is to make CBWC from the 2d histogram, like a muppet.
def apply_cbwc(refmult, protons, binsX, pro_int_cumus, ref_cuts):
    pro_cbwc = np.empty((4, len(binsX) - 1))
    pro_len = np.zeros(len(binsX[:-1]))
    for i in range(len(binsX[:-1])):
        index = refmult == i
        arr = protons[index]
        pro_len[i] = len(arr)
    for i in range(4):
        pro_cbwc[i] = np.multiply(pro_int_cumus[i], pro_len)
    pro_len_tot = [[], [], [], []]
    pro_len_err = [[], [], [], []]
    for i in range(len(ref_cuts) - 1):
        for j in range(4):
            index_nan = ~np.isnan(pro_cbwc[j][ref_cuts[i]:ref_cuts[i + 1]])
            cbwc = np.sum(pro_cbwc[j][ref_cuts[i]:ref_cuts[i + 1]][index_nan])
            nsum = np.sum(pro_len[ref_cuts[i]:ref_cuts[i + 1]][index_nan])
            pro_len_tot[j].append(np.divide(cbwc, nsum))
            pro_len_err[j].append(np.std(pro_cbwc[j][ref_cuts[i]:ref_cuts[i + 1]][index_nan])/np.sqrt(nsum))
    for i in range(4):
        index_nan = ~np.isnan(pro_cbwc[i][ref_cuts[-1]:])
        cbwc = np.sum(pro_cbwc[i][ref_cuts[-1]:][index_nan])
        nsum = np.sum(pro_len[ref_cuts[-1]:][index_nan])
        pro_len_tot[i].append(np.divide(cbwc, nsum))
        pro_len_err[i].append(np.divide(np.std(pro_cbwc[i][ref_cuts[-1]:][index_nan]), np.sqrt(nsum)))
    pro_len_tot = np.array(pro_len_tot)
    pro_len_err = np.array(pro_len_err)

    pro_len_tot_comp = np.copy(pro_len_tot)
    pro_len_tot_comp[1] = np.divide(pro_len_tot[1], pro_len_tot[0])
    pro_len_tot_comp[2] = np.divide(pro_len_tot[2], pro_len_tot[1])
    pro_len_tot_comp[3] = np.divide(pro_len_tot[3], pro_len_tot[1])

    return pro_len_tot, pro_len_tot_comp, pro_len_err


# This is to make CBWC by proton and refmult distributions.
def apply_cbwc_protons(refmult, protons, ref_cuts):
    pro_cbwc = np.empty((4, len(ref_cuts)))
    pro_cbwc_err = np.empty((4, len(ref_cuts)))
    for i in range(len(ref_cuts) - 1):
        index = (refmult >= ref_cuts[i]) & (refmult < ref_cuts[i + 1])
        pro = protons[index]
        pro_len_tot = len(pro)
        refq = np.unique(refmult[index])
        pro_mean = 0
        pro_mean_err = np.zeros(len(refq))
        pro_var = 0
        pro_var_err = np.zeros(len(refq))
        pro_skew = 0
        pro_skew_err = np.zeros(len(refq))
        pro_kurt = 0
        pro_kurt_err = np.zeros(len(refq))
        for j in range(len(refq)):
            index_j = refmult[index] == refq[j]
            pro_mean += np.mean(pro[index_j]) * len(pro[index_j])
            pro_mean_err[j] = np.mean(pro[index_j])
            pro_var += np.var(pro[index_j]) * len(pro[index_j])
            pro_var_err[j] = np.var(pro[index_j])
            pro_skew += (skew(pro[index_j]) * np.power(np.sqrt(np.var(pro[index_j])), 3)) * len(pro[index_j])
            pro_skew_err[j] = (skew(pro[index_j]) * np.power(np.sqrt(np.var(pro[index_j])), 3))
            pro_kurt += (kurtosis(pro[index_j]) * np.power(np.var(pro[index_j]), 2)) * len(pro[index_j])
            pro_kurt_err[j] = (kurtosis(pro[index_j]) * np.power(np.var(pro[index_j]), 2))
        if pro_len_tot > 0:
            pro_cbwc[0][i] = pro_mean / pro_len_tot
            pro_cbwc[1][i] = pro_var / pro_len_tot
            pro_cbwc[2][i] = pro_skew / pro_len_tot
            pro_cbwc[3][i] = pro_kurt / pro_len_tot
            pro_cbwc_err[0][i] = np.std(pro_mean_err) / np.sqrt(len(pro_mean_err))
            if ((len(pro_var_err) - 1) != 0) & ((len(pro_var_err) - 2) != 0) & ((len(pro_var_err) - 3) != 0):
                n = len(pro_var_err)
                pro_cbwc_err[1][i] = np.var(pro_var_err) * np.sqrt(2 / (n - 1))
                pro_cbwc_err[2][i] = (6 * n * (n - 1)) / ((n + 1) * (n - 2) * (n + 3))
                pro_cbwc_err[3][i] = (4 * ((n ** 2) - 1) * pro_cbwc_err[2][i]) / ((n - 3) * (n + 5))
    index = (refmult >= ref_cuts[-1])
    pro = protons[index]
    pro_len_tot = len(pro)
    refq = np.unique(refmult[index])
    pro_mean = 0
    pro_mean_err = np.zeros(len(refq))
    pro_var = 0
    pro_var_err = np.zeros(len(refq))
    pro_skew = 0
    pro_skew_err = np.zeros(len(refq))
    pro_kurt = 0
    pro_kurt_err = np.zeros(len(refq))
    for j in range(len(refq)):
        index_j = refmult[index] == refq[j]
        pro_mean += np.mean(pro[index_j]) * len(pro[index_j])
        pro_mean_err[j] = np.mean(pro[index_j])
        if ((len(pro_var_err) - 1) != 0) & ((len(pro_var_err) - 2) != 0) & ((len(pro_var_err) - 3) != 0):
            n = len(pro_var_err)
            pro_var_err[j] = np.var(pro_var_err) * np.sqrt(2 / (n - 1))
            pro_skew_err[j] = (6 * n * (n - 1)) / ((n + 1) * (n - 2) * (n + 3))
            pro_kurt_err[j] = (4 * ((n ** 2) - 1) * pro_skew_err[j]) / ((n - 3) * (n + 5))
        pro_var += np.var(pro[index_j]) * len(pro[index_j])
        # pro_var_err[j] = np.var(pro[index_j])
        pro_skew += (skew(pro[index_j]) * np.power(np.sqrt(np.var(pro[index_j])), 3)) * len(pro[index_j])
        # pro_skew_err[j] = (skew(pro[index_j]) * np.power(np.sqrt(np.var(pro[index_j])), 3))
        pro_kurt += (kurtosis(pro[index_j]) * np.power(np.var(pro[index_j]), 2)) * len(pro[index_j])
        # pro_kurt_err[j] = (kurtosis(pro[index_j]) * np.power(np.var(pro[index_j]), 2))
    pro_cbwc[0][-1] = pro_mean / pro_len_tot
    pro_cbwc[1][-1] = pro_var / pro_len_tot
    pro_cbwc[2][-1] = pro_skew / pro_len_tot
    pro_cbwc[3][-1] = pro_kurt / pro_len_tot
    pro_cbwc_err[0][-1] = np.std(pro_mean_err) / np.sqrt(len(pro_mean_err))
    pro_cbwc_err[1][-1] = np.var(pro_var_err)*np.sqrt(2/(len(pro_var_err)-1))
    pro_cbwc_err[2][-1] = (6 * len(pro_skew_err) * (len(pro_skew_err)-1)) / ((len(pro_skew_err) + 1) * (len(pro_skew_err) - 2) * (len(pro_skew_err) + 5))
    pro_cbwc_err[3][-1] = (4 * ((len(pro_skew_err) ** 2) -1) * pro_cbwc_err[2][-1]) / ((pro_cbwc_err[2][-1] -3) * (pro_cbwc_err[2][-1] + 5))

    pro_cbwc_comp = np.copy(pro_cbwc)
    pro_cbwc_comp[1] = pro_cbwc[1] / pro_cbwc[0]
    pro_cbwc_comp[2] = pro_cbwc[2] / pro_cbwc[1]
    pro_cbwc_comp[3] = pro_cbwc[3] / pro_cbwc[1]

    return pro_cbwc, pro_cbwc_comp, pro_cbwc_err


# Making cumulant distributions like a moron.
def int_cumulants(binsX, binsY, counter):
    # Now to make a distribution by RefMult of our cut correlations and get the cumuants.
    pro_int = []
    pro_len = []
    pro_int_cumus = [[], [], [], []]
    for i in binsX.astype('int')[:-1]:
        pro_int.append([])
        arr = counter[i, :]
        for j in binsY.astype('int')[:-1]:
            for k in range(int(arr[j])):
                pro_int[i].append(j)
        pro_int_cumus[0].append(np.mean(pro_int[i]))
        pro_int_cumus[1].append(np.var(pro_int[i]))
        pro_int_cumus[2].append(skew(np.array(pro_int[i])) *
                                np.power(np.sqrt(np.var(pro_int[i])), 3))
        pro_int_cumus[3].append(kurtosis(pro_int[i])
                                * np.power(np.var(pro_int[i]), 2))
        pro_len.append(len(pro_int[i]))
    pro_int_cumus = np.array(pro_int_cumus)

    # And again, but for the regular ratios we see.
    pro_int_cumcomp = np.copy(pro_int_cumus)
    pro_int_cumcomp[1] = np.divide(pro_int_cumus[1], pro_int_cumus[0])
    pro_int_cumcomp[2] = np.divide(pro_int_cumus[2], pro_int_cumus[1])
    pro_int_cumcomp[3] = np.divide(pro_int_cumus[3], pro_int_cumus[1])

    return pro_int_cumus, pro_int_cumcomp


# Making cumulant distributions by integer refmult.
def int_cumulants_protons(refmult, protons):
    # Now to make a distribution by RefMult of our cut correlations and get the cumuants.
    pro_int = []
    pro_int_cumus = [[], [], [], []]
    pro_int_errs = [[], [], [], []]
    for i in np.unique(refmult):
        index = refmult == i
        arr = protons[index]
        # Quantities we'll need
        std = np.std(arr)
        n = len(arr)
        # Calculate the mean and error
        mean = np.mean(arr)
        mean_err = std / np.sqrt(n)
        # Calculate the variance and error
        var = np.var(arr)
        # Calculate the skewness and kurtosis, with error
        skewness = skew(arr) * np.power(np.sqrt(np.var(arr)), 3)
        kurt = kurtosis(arr) * np.power(np.var(arr), 2)
        var_err = 0
        skew_err = 0
        kurt_err = 0
        if ((n - 1) != 0) & ((n - 2) != 0) & ((n - 3) != 0):
            var_err = var * np.sqrt(2 / (n - 1))
            skew_err = (6 * n * (n - 1)) / ((n + 1) * (n - 2) * (n + 3))
            kurt_err = (4 * ((n ** 2) - 1) * skew_err) / ((n - 3) * (n + 5))

        pro_int_cumus[0].append(mean)
        pro_int_errs[0].append(mean_err)
        pro_int_cumus[1].append(var)
        pro_int_errs[1].append(var_err)
        pro_int_cumus[2].append(skewness)
        pro_int_errs[2].append(skew_err)
        pro_int_cumus[3].append(kurt)
        pro_int_errs[3].append(kurt_err)

    pro_int_cumus = np.array(pro_int_cumus)
    pro_int_errs = np.array(pro_int_errs)

    # And again, but for the regular ratios we see.
    pro_int_cumcomp = np.copy(pro_int_cumus)
    pro_int_cumcomp[1] = np.divide(pro_int_cumus[1], pro_int_cumus[0])
    pro_int_cumcomp[2] = np.divide(pro_int_cumus[2], pro_int_cumus[1])
    pro_int_cumcomp[3] = np.divide(pro_int_cumus[3], pro_int_cumus[1])

    return pro_int_cumus, pro_int_cumcomp, pro_int_errs


# 1D histograms of the proton distributions.
def proton_histos_1d(refmult, protons, ref_raw, pro_raw, ref_cuts, ref_deltas):
    fig, ax = plt.subplots(3, 4, figsize=(16, 9), constrained_layout=True)
    for i in range(3):
        for j in range(4):
            x = 4 * i + j
            if x > 8:
                continue
            index = (refmult >= ref_cuts[x]) & (refmult < ref_cuts[x + 1])
            ax[i, j].hist(protons[index], bins=50, range=(0, 50), histtype='step', density=True,
                          label="Truncated Runs")
            index = (ref_raw >= ref_cuts[x]) & (ref_raw < ref_cuts[x + 1])
            ax[i, j].hist(pro_raw[index], bins=50, range=(0, 50), histtype='step', density=True,
                          color='r', alpha=0.5, label="All Runs")
            ax[i, j].set_yscale('log')
            ax[i, j].set_title(ref_deltas[x])
            ax[i, j].set_xlabel("Protons")
            ax[i, j].set_ylabel(r"$\frac{dN}{dProtons}$")
    index = refmult >= ref_cuts[-1]
    ax[2, 1].hist(protons[index], bins=50, range=(0, 50), histtype='step', density=True,
                  label="Truncated Runs")
    index = ref_raw >= ref_cuts[-1]
    ax[2, 1].hist(pro_raw[index], bins=50, range=(0, 50), histtype='step', density=True,
                  color='r', alpha=0.5, label="All Runs")
    ax[2, 1].set_yscale('log')
    ax[2, 1].set_title(ref_deltas[-1])
    ax[2, 1].set_xlabel("Protons")
    ax[2, 1].set_ylabel(r"$\frac{dN}{dProtons}$")
    ax[0, 0].legend()
    ax[2, 2].set_axis_off()
    ax[2, 3].set_axis_off()
    plt.show()
    plt.close()
