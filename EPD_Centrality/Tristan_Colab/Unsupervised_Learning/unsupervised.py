#
# \brief Unsupervised learning methods to find correlations between EPD rings and centrality
#
# \author Skipper KAgamaster
# \date 02/04/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
# /

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import uproot as up
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy import linalg
from sklearn import mixture
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils.generic_utils import get_custom_objects
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras.callbacks import EarlyStopping


class GetData:
    def __init__(self):
        self.rings = None
        self.tpc = None
        self.b = None
        self.ref_cuts = None

    def data_in(self, file_loc):
        file = up.open(file_loc)
        data_len = int(len(file['ring_sums'].member("fElements")) / 16)
        self.rings = np.reshape(file["ring_sums"].member("fElements"), (16, data_len)).T
        self.tpc = file["tpc_multiplicity"].member("fElements")
        self.b = file["impact_parameter"].member("fElements")
        self.ref_cuts = np.percentile(self.tpc, [95, 90, 80, 70, 60, 50, 40, 30])
        print(self.ref_cuts)


def raw_sums_display(rings, tpc, b):
    fig, ax = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    counter, bins_y, bins_x = np.histogram2d(tpc, b, bins=100)
    tpc_b = ax[0, 0].pcolormesh(bins_x, bins_y, counter, cmap='jet', norm=colors.LogNorm())
    ax[0, 0].set_xlabel("b (fm)", fontsize=10)
    ax[0, 0].set_ylabel("RefMult3", fontsize=10)
    ax[0, 0].set_title("RefMult3 vs b", fontsize=20)
    fig.colorbar(tpc_b, ax=ax[0, 0])

    counter, bins_y, bins_x = np.histogram2d(np.sum(rings, axis=1), b, bins=100)
    epd_b = ax[0, 1].pcolormesh(bins_x, bins_y, counter, cmap='jet', norm=colors.LogNorm())
    ax[0, 1].set_xlabel("b", fontsize=10)
    ax[0, 1].set_ylabel("EPD rings (sum)", fontsize=10)
    ax[0, 1].set_title("EPD Rings vs b", fontsize=20)
    fig.colorbar(epd_b, ax=ax[0, 1])

    counter, bins_y, bins_x = np.histogram2d(rings[:, 1], b, bins=100)
    epd_b_inner = ax[1, 0].pcolormesh(bins_x, bins_y, counter, cmap='jet', norm=colors.LogNorm())
    ax[1, 0].set_xlabel("b", fontsize=10)
    ax[1, 0].set_ylabel("EPD ring 2", fontsize=10)
    ax[1, 0].set_title("EPD Inner Ring vs b", fontsize=20)
    ax[1, 0].set_ylim(0, 75)
    fig.colorbar(epd_b, ax=ax[1, 0])

    counter, bins_x, bins_y = np.histogram2d(b, rings[:, 14], bins=100)
    epd_b_outer = ax[1, 1].pcolormesh(bins_x, bins_y, counter, cmap='jet', norm=colors.LogNorm())
    ax[1, 1].set_xlabel("b", fontsize=10)
    ax[1, 1].set_ylabel("EPD ring 15", fontsize=10)
    ax[1, 1].set_title("EPD Outer Ring vs b", fontsize=20)
    ax[1, 1].set_ylim(0, 75)
    fig.colorbar(epd_b, ax=ax[1, 1])
    plt.show()


# This is a KMeans model, with clusters n.
def model_kmeans(data, n_clusters=5, n_init=10, max_iter=300):
    model = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)
    model.fit(data)
    predictions = model.predict(data)
    return predictions


# KMeans Minibach model, with clusters n.
def model_kmeans_minbatch(data, n_clusters=10, n_init=10, max_iter=300):
    model = MiniBatchKMeans(init='random', n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)
    model.fit(data)
    predictions = model.predict(data)
    return predictions


# Gausian Mixture Model (GMM; form of k-means).
def model_gmm(data, n_clusters=10, n_init=10, cov_type='diag', max_iter=10):
    model = mixture.GaussianMixture(n_components=n_clusters, covariance_type=cov_type,
                                    n_init=n_init, max_iter=max_iter)
    model.fit(data)
    predictions = model.predict(data)
    return predictions


# Display for the rings vs b with KMeans clustering colours.
def kmeans_display(kmeans, rings, b, cmap="jet", m_type='KMeans'):  # n is for the number of clusters.
    n = len(np.unique(kmeans))
    cmap_ = plt.cm.get_cmap(cmap, n)
    fig, ax = plt.subplots(4, 4, figsize=(12, 9), constrained_layout=True)
    for i in range(4):
        for j in range(4):
            x = i * 4 + j
            ax[i, j].scatter(b, rings[:, x], c=kmeans, cmap=cmap_, alpha=0.5)
            ax[i, j].set_xlabel("b (fm)", fontsize=10)
            ax[i, j].set_ylabel("EPD Ring {}".format(x + 1), fontsize=10)
    fig.suptitle("{} ({}) Sorting by Ring".format(m_type, n), fontsize=20)
    plt.show()


def b_distributions_kmeans(vals, b, kmeans, cmap="jet", m_type='KMeans'):
    n = len(np.unique(kmeans))
    cmap_ = plt.cm.get_cmap(cmap, n)
    count = 0
    for i in vals:
        index = np.where(kmeans == i)
        plt.hist(b[index], bins=100, histtype='step', range=(0, 16), density=True, label=i, color=cmap_(count))
        count += 1
    plt.xlabel("b (fm)", fontsize=15)
    plt.ylabel("dN/db", fontsize=15)
    plt.title("b Distributions for {}_{}".format(m_type, n), fontsize=20)
    plt.legend()
    plt.show()


def tpc_b_plot(tpc, b, ref_cuts):
    plt.hist(b[tpc >= ref_cuts[0]], bins=100, histtype='step', range=(0, 16), density=True, label="0")
    for i in range(len(ref_cuts)-1):
        index = (tpc >= ref_cuts[i+1]) & (tpc < ref_cuts[i])
        plt.hist(b[index], bins=100, histtype='step', range=(0, 16), density=True, label="{}".format(i+1))
    plt.hist(b[tpc < ref_cuts[len(ref_cuts)-1]], bins=100, histtype='step',
             range=(0, 16), density=True, label="{}".format(len(ref_cuts)))
    plt.xlabel("b (fm)", fontsize=15)
    plt.ylabel("dN/db", fontsize=15)
    plt.title("b Distributions for RefMult3", fontsize=20)
    plt.legend()
    plt.show()


def tpc_split_kmeans(b, tpc, kmeans, n, cmap="jet"):
    cmap_ = plt.cm.get_cmap(cmap, n)
    plt.scatter(b, tpc, c=kmeans, cmap="jet", alpha=0.5)
    plt.xlabel("b (fm)", fontsize=15)
    plt.ylabel("RefMult3", fontsize=15)
    plt.title("RefMult3 vs b, Color from KMeans_{}".format(n), fontsize=20)
    for i in range(n):
        plt.scatter(0, i, s=0.001, label="{}".format(i), c=cmap_(i))
    plt.legend(markerscale=300)
    plt.show()


def kmeans_ringvs_display(rings, kmeans, n=10, cmap="jet"):
    cmap_ = plt.cm.get_cmap(cmap, n)
    fig, ax = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for i in range(2):
        for j in range(2):
            ax[i, j].scatter(rings[:, i*4+j*2], rings[:, 14-j*2], c=kmeans, cmap=cmap_)
            ax[i, j].set_xlabel("Ring {}".format(i*4 + j*2), fontsize=10)
            ax[i, j].set_ylabel("Ring {}".format(14-j*2), fontsize=10)
    fig.suptitle("KMeans ({}) Ring by Ring Comparisons".format(n), fontsize=20)
    plt.show()
