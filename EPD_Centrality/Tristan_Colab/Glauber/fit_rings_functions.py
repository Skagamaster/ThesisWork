import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
from scipy.optimize import curve_fit
from numpy.random import default_rng
rng = default_rng()


def gmc_dist_generator_old(n_coll, n_part, n, p, alpha):
    nbd = rng.negative_binomial(n, p, int(1e6))
    gmc = np.add(alpha * n_coll, (1 - alpha) * n_part).astype("int")
    n_pp = np.empty(len(gmc))
    for i in range(len(gmc)):
        n_pp[i] = np.sum(rng.choice(nbd, gmc[i]))
    return n_pp


def gmc_dist_generator(n_coll, n_part, n, p, alpha):
    gmc = np.add(alpha * n_coll, (1 - alpha) * n_part).astype("int") * n
    gmc = gmc[gmc != 0]
    n_pp = rng.negative_binomial(gmc, p)
    return n_pp


def nbd_generator(n, p):
    nbd = rng.negative_binomial(n, p, int(1e6))
    return nbd


def gmc_short(nbd, alpha, n_coll, n_part):
    gmc = np.add(alpha * n_coll, (1 - alpha) * n_part).astype("int")
    n_pp = np.empty(len(gmc))
    for i in range(len(gmc)):
        n_pp[i] = np.sum(rng.choice(nbd, gmc[i]))
    return n_pp


def load_data(df_file=r"C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results\BigList\out_all.pkl",
              pred_loc=r"C:\Users\dansk\Documents\Thesis\2021_DNP_Work\ML_fits\9_2021\\",
              ring_set=np.array((7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)),
              species_energy_n="197Au_197Au_14.6_1000000",
              gmc_file=r"C:\Users\dansk\Documents\Thesis\Tristan"):
    df = pd.read_pickle(df_file)
    ring_sum = np.zeros(len(df))
    # Sum up all the outer ring sums (both sides of the EPD).
    for i in ring_set:
        ring_sum = ring_sum + df["ring{}".format(i)].to_numpy()
    ring_sum = np.round(ring_sum).astype('int')
    # Now get the Glauber MC portions.
    n_part = np.load(r'{}\Npart_{}.npy'.format(gmc_file, species_energy_n), allow_pickle=True)
    n_coll = np.load(r'{}\Ncoll_{}.npy'.format(gmc_file, species_energy_n), allow_pickle=True)
    # Now to get ML predictions for various schemes.
    pred_linear = np.load(pred_loc+"linearpredictions.npy", allow_pickle=True)
    pred_relu = np.load(pred_loc+"relupredictions.npy", allow_pickle=True)
    pred_swish = np.load(pred_loc+"swishpredictions.npy", allow_pickle=True)
    pred_mish = np.load(pred_loc+"mishpredictions.npy", allow_pickle=True)
    predictions = np.array((pred_linear, pred_relu, pred_swish, pred_mish))
    refmult = df['refmult'].to_numpy()
    return ring_sum, n_coll, n_part, predictions, refmult


def load_data_all(pred_loc=r"C:\Users\dansk\Documents\Thesis\2021_DNP_Work\ML_fits\9_2021\\",
                  species_energy_n="197Au_197Au_14.5_100000",
                  gmc_file=r"C:\Users\dansk\Documents\Thesis\Tristan"):
    # First get the Glauber MC portions.
    n_part = np.load(r'{}\Npart_{}.npy'.format(gmc_file, species_energy_n), allow_pickle=True)
    n_coll = np.load(r'{}\Ncoll_{}.npy'.format(gmc_file, species_energy_n), allow_pickle=True)
    # Now to get ML predictions for various schemes.
    pred_linear = np.load(pred_loc + "linearpredictions.npy", allow_pickle=True)
    pred_relu = np.load(pred_loc + "relupredictions.npy", allow_pickle=True)
    pred_swish = np.load(pred_loc + "swishpredictions.npy", allow_pickle=True)
    pred_mish = np.load(pred_loc + "mishpredictions.npy", allow_pickle=True)
    # And now the remaining quantities.
    ring_sum = np.load(pred_loc+"ring_sum.npy", allow_pickle=True)
    ring_sum_outer = np.load(pred_loc+"ring_sum_outer.npy", allow_pickle=True)
    refmult = np.load(pred_loc+"refmult3.npy", allow_pickle=True)
    df = pd.DataFrame(np.array((refmult, ring_sum, ring_sum_outer, pred_linear, pred_relu, pred_swish, pred_mish)).T,
                      columns=["refmult", "ring_sum", "ring_sum_outer", "pred_linear",
                               "pred_relu", "pred_swish", "pred_mish"])

    return n_part, n_coll, df
