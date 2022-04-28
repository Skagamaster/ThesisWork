import unsupervised as usp
import numpy as np
import matplotlib.pyplot as plt

# Importing the 7.7 GeV UrQMD data from Tristan
data = usp.GetData()
data.data_in(r"C:\Users\dansk\Downloads\simulated_data.root")
rings = data.rings  # EPD ring by ring data; 16 rings (no discernment of east and west)
tpc = data.tpc  # RefMult3
b = data.b  # Impact parameter
ref_cuts = data.ref_cuts  # Quantiles for RefMult3

# Build the KMeans model (unsupervised learning).
n = 3  # Number of clusters
n_init = 50  # Number of different centroid seeds to try (picks the "best" one)
max_iter = 10000  # Max iterations for each kmeans algorithm run
cmap = plt.cm.get_cmap("jet", n)  # Colour selection

# Train the KMeans model on all the rings, inner rings, and outer rings.
kmeans = usp.model_kmeans(rings, n, n_init, max_iter).astype(int)
# kmeans_low = usp.model_kmeans(rings[:, :7], n, n_init, max_iter).astype(int)
# kmeans_high = usp.model_kmeans(rings[:, 7:], n, n_init, max_iter).astype(int)
gmm = usp.model_gmm(rings, n, n_init, 'spherical', max_iter)
'''
# Visualisation of the clusters, rings vs b.
usp.kmeans_display(kmeans, rings, b, cmap)
usp.kmeans_display(kmeans_low, rings, b, cmap)
usp.kmeans_display(kmeans_high, rings, b, cmap)
# All rings compared using KMeans clustering.
usp.kmeans_ringvs_display(rings, kmeans, n)
usp.kmeans_ringvs_display(rings, kmeans_low, n)
usp.kmeans_ringvs_display(rings, kmeans_high, n)
'''
# Clusters
vals = np.unique(kmeans)
# vals_low = np.unique(kmeans_low)
# vals_high = np.unique(kmeans_high)
vals_gmm = np.unique(gmm)

# b distributions for each set of clusters/quantiles (EPD/TPC, respectively).
usp.b_distributions_kmeans(vals, b, kmeans, m_type='KMeans')
# usp.b_distributions_kmeans(vals_low, b, kmeans_low, m_type='KMeans')
# usp.b_distributions_kmeans(vals_high, b, kmeans_high, m_type='KMeans')
usp.b_distributions_kmeans(vals_gmm, b, gmm, m_type='GMM_spherical')
usp.tpc_b_plot(tpc, b, ref_cuts)

# Where the KMeans clusters live in TPC/b space.
usp.tpc_split_kmeans(b, tpc, gmm, n)
# usp.tpc_split_kmeans(b, tpc, kmeans_low, n)
# usp.tpc_split_kmeans(b, tpc, kmeans_high, n)

most_central = input("Which cluster was the central cluster?")
most_central = int(most_central)
mid_central = input("Which cluster was in the middle?")
mid_central = int(mid_central)
peripheral = input("Which cluster was most peripheral?")
peripheral = int(peripheral)

cent_ind = np.hstack(np.where(gmm == most_central))
cent_rings = rings[cent_ind]
cent_b = b[cent_ind]
cent_tpc = tpc[cent_ind]
cent_kmeans = usp.model_gmm(cent_rings, 3, n_init, 'spherical').astype(int)
cent_vals = np.unique(cent_kmeans)
usp.b_distributions_kmeans(cent_vals, cent_b, cent_kmeans, m_type='GMM_spherical')
usp.tpc_split_kmeans(cent_b, cent_tpc, cent_kmeans, 3)

mid_ind = np.hstack(np.where(gmm == mid_central))
mid_rings = rings[mid_ind]
mid_b = b[mid_ind]
mid_tpc = tpc[mid_ind]
mid_kmeans = usp.model_gmm(mid_rings, 3, n_init, 'spherical').astype(int)
mid_vals = np.unique(mid_kmeans)
usp.b_distributions_kmeans(mid_vals, mid_b, mid_kmeans, m_type='GMM_spherical')
usp.tpc_split_kmeans(mid_b, mid_tpc, mid_kmeans, 3)

per_ind = np.hstack(np.where(gmm == peripheral))
per_rings = rings[per_ind]
per_b = b[per_ind]
per_tpc = tpc[per_ind]
per_kmeans = usp.model_gmm(per_rings, 3, n_init, 'spherical').astype(int)
per_vals = np.unique(per_kmeans)
usp.b_distributions_kmeans(per_vals, per_b, per_kmeans, m_type='GMM_spherical')
usp.tpc_split_kmeans(per_b, per_tpc, per_kmeans, 3)

all_b = np.hstack((cent_b, mid_b, per_b))
all_kmeans = np.hstack((cent_kmeans, mid_kmeans, per_kmeans))
all_vals = np.hstack((cent_vals, mid_vals, per_vals))
usp.b_distributions_kmeans(all_vals, all_b, all_kmeans, m_type='GMM_spherical_splits')

MLP_rings = np.load(r"C:\Users\dansk\Documents\Thesis\Tristan\Unsupervised_Learning\Supervised_Data\MLP_predictions.npy", allow_pickle=True)
ring_cuts = np.percentile(MLP_rings, [95, 90, 80, 70, 60, 50, 40, 30])
ring_cuts = np.percentile(MLP_rings, [5, 10, 20, 30, 40, 50, 60, 70])
usp.tpc_b_plot(MLP_rings, b, ring_cuts[::-1])

'''
plt.hist(b[MLP_rings < ring_cuts[len(ring_cuts) - 1]], bins=100, histtype='step', range=(0, 16), density=True, label="0")
for i in range(len(ring_cuts) - 1):
    index = (MLP_rings < ring_cuts[i + 1]) & (MLP_rings >= ring_cuts[i])
    plt.hist(b[index], bins=100, histtype='step', range=(0, 16), density=True, label="{}".format(i+1))
plt.xlabel("b (fm)", fontsize=15)
plt.ylabel("dN/db", fontsize=15)
plt.title("b Distributions for EPD Rings (MLP)", fontsize=20)
plt.legend()
plt.show()
'''
