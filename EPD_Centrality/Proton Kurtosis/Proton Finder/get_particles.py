#
# \brief Unsupervised learning methods to find protons
#
# \author Skipper KAgamaster
# \date 03/10/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
# /

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import uproot as up
import awkward as ak
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy import linalg
from sklearn import mixture
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


def rapidity_awk(p_z):
    """Takes proton momentum and returns rapidity
    Args:
        p_z (str): z component of the momentum
    """
    e_p_p = np.power(np.add(0.9382720813 ** 2, np.power(p_z, 2)), 1 / 2)
    e_m_p = np.subtract(0.9382720813 ** 2, np.power(p_z, 2))
    e_m_p = ak.where(e_m_p < 0.0, 0.0, e_m_p)  # to avoid imaginary numbers
    e_m_p = np.power(e_m_p, 1 / 2)
    e_m_p = ak.where(e_m_p == 0.0, 1e-10, e_m_p)  # to avoid infinities
    y = np.multiply(np.log(np.divide(e_p_p, e_m_p)), 1 / 2)
    return y


def charge_list(a):
    """Takes nHitsFit and returns charge as +- 1
       This simply normalises nHitsFit and returns the sign.
       TODO Is there just a way to get sign in numpy?
    Args:
        a (float): nHitsFit
    """
    a = abs(a) / a
    return a


class ParticleFinder:

    def __init__(self):
        self.data1 = None
        self.data2 = None
        self.data3 = None
        self.kmeans = None
        self.gmm = None
        self.p_t = None
        self.p_g = None
        self.beta = None
        self.dedx = None
        self.charge = None
        self.m_squared = None

    def data_in(self, file_loc, r_cut=2.0, z_cut=30.0, dedx_cut=5,
                nhits_cut=20, fit_max_ratio=0.52, dca_cut=1.0,
                rapidity_cut=0.5):
        """Loads the data from the specified root file into memory,
        allowing the model to be created

        Args:
            file_loc (str): The path to the root file containing the summary
                             of the ring values and the impact parameter/refmult
            r_cut (float): Where to cut on transverse vertex
            z_cut (float): Where to cut on z vertex
            dedx_cut (float): To get rid of noise in dE/dx
            nhits_cut (int): Minimum nHitsFit
            fit_max_ratio (float): Minimum ratio of nHitsFit/nHitsMax
            dca_cut (float): Max DCA
            rapidity_cut (float): Max rapidity (for protons)
        """
        file = up.open(file_loc)
        try:
            data = file["PicoDst"]

        except ValueError:  # Skip empty picos.
            print("Pico is empty.")  # Identifies the misbehaving file.
        except KeyError:  # Skip non empty picos that have no data.
            print("Pico has no data.")  # Identifies the misbehaving file.
        except Exception as e:  # For any other issues that might pop up.
            print(e.__class__, "occurred.")

        try:
            # Perform some QA, then build the input vector for the ML fit.

            # Event level quantities and cuts.
            v_x = ak.to_numpy(ak.flatten(data["Event"]["Event.mPrimaryVertexX"].array()))
            v_y = ak.to_numpy(ak.flatten(data["Event"]["Event.mPrimaryVertexY"].array()))
            v_z = ak.to_numpy(ak.flatten(data["Event"]["Event.mPrimaryVertexZ"].array()))
            v_r = np.power((np.power(v_x, 2) + np.power(v_y, 2)), (1 / 2))
            zdc_x = ak.to_numpy(ak.flatten(data["Event"]["Event.mZDCx"].array()))

            event_cuts = ((v_r <= r_cut) & (np.absolute(v_z) <= z_cut))
            v_x, v_y, v_z, v_r = v_x[event_cuts], v_y[event_cuts], v_z[event_cuts], v_r[event_cuts]
            zdc_x = zdc_x[event_cuts]

            ref_mult_3 = ak.to_numpy(ak.flatten(data["Event"]["Event.mRefMult3PosEast"].array() +
                                                data["Event"]["Event.mRefMult3PosWest"].array() +
                                                data["Event"]["Event.mRefMult3NegEast"].array() +
                                                data["Event"]["Event.mRefMult3NegWest"].array())[event_cuts])
            tof_mult = ak.to_numpy(ak.flatten(data["Event"]["Event.mbTofTrayMultiplicity"].array())[event_cuts])
            tof_match = ak.to_numpy(ak.flatten(data["Event"]["Event.mNBTOFMatch"].array())[event_cuts])

            # Track level quantities. I will ToF match all quantities.
            tof_pid = data["BTofPidTraits"]["BTofPidTraits.mTrackIndex"].array()[event_cuts]

            p_x = data["Track"]["Track.mGMomentumX"].array()[event_cuts][tof_pid]
            p_y = data["Track"]["Track.mGMomentumY"].array()[event_cuts][tof_pid]
            p_y = ak.where(p_y == 0.0, 1e-10, p_y)  # to avoid infinities
            p_z = data["Track"]["Track.mGMomentumZ"].array()[event_cuts][tof_pid]
            p_t = np.power((np.power(p_x, 2) + np.power(p_y, 2)), (1 / 2))
            p_g = np.power((np.power(p_x, 2) + np.power(p_y, 2) + np.power(p_z, 2)), (1 / 2))

            eta = np.arcsinh(np.divide(p_z, np.sqrt(np.add(np.power(p_x, 2), np.power(p_y, 2)))))
            rapidity = rapidity_awk(p_z)

            dca_x = data["Track"]["Track.mOriginX"].array()[event_cuts][tof_pid] - v_x
            dca_y = data["Track"]["Track.mOriginY"].array()[event_cuts][tof_pid] - v_y
            dca_z = data["Track"]["Track.mOriginZ"].array()[event_cuts][tof_pid] - v_z
            dca = np.power((np.power(dca_x, 2) + np.power(dca_y, 2) + np.power(dca_z, 2)), (1 / 2))

            n_hits_dedx = data["Track"]["Track.mNHitsDedx"].array()[event_cuts][tof_pid]
            n_hits_fit = data["Track"]["Track.mNHitsFit"].array()[event_cuts][tof_pid]
            n_hits_max = data["Track"]["Track.mNHitsMax"].array()[event_cuts][tof_pid]
            n_hits_max = ak.where(n_hits_max == 0, 1e-10, n_hits_max)  # to avoid infinities

            # Scaling for beta is a STAR thing; see StPicoBTofPidTraits.h.
            beta = data["BTofPidTraits"]["BTofPidTraits.mBTofBeta"].array()[event_cuts] / 20000.0
            dedx = data["Track"]["Track.mDedx"].array()[event_cuts][tof_pid]
            n_sigma_proton = data["Track"]["Track.mNSigmaProton"].array()[event_cuts][tof_pid]

            # Now for some track level cuts.
            track_cuts = ((n_hits_dedx > dedx_cut) & (np.absolute(n_hits_fit) > nhits_cut) &
                          (np.divide(np.absolute(n_hits_fit), n_hits_max) > fit_max_ratio) &
                          (dca < dca_cut) & (np.absolute(rapidity) <= rapidity_cut))
            p_x, p_y, p_z, p_t, p_g, eta, rapidity, \
                dca, n_hits_dedx, n_hits_fit, n_hits_max, beta, \
                dedx, n_sigma_proton, = p_x[track_cuts], p_y[track_cuts], \
                                        p_z[track_cuts], p_t[track_cuts], \
                                        p_g[track_cuts], eta[track_cuts], \
                                        rapidity[track_cuts], dca[track_cuts], \
                                        n_hits_dedx[track_cuts], n_hits_fit[track_cuts], \
                                        n_hits_max[track_cuts], beta[track_cuts], \
                                        dedx[track_cuts], n_sigma_proton[track_cuts]

            # Now for another event level cut (on RefMults). This is
            # for pileup and unknown nonconformity rejection.
            beta_eta1_match = ak.where((beta > 0.1) & (np.absolute(eta) < 1.0) &
                                       (np.absolute(dca < 3.0) & (np.absolute(n_hits_fit) > 10)),
                                       1, 0)
            beta_eta1 = ak.sum(beta_eta1_match, axis=-1)
            event_cuts2 = ((tof_mult >= (1.22 * ref_mult_3 - 24.29)) &
                           (tof_mult <= (1.95 * ref_mult_3 + 75)) &
                           (tof_match >= (0.379 * ref_mult_3 - 8.6)) &
                           (tof_match <= (0.631 * ref_mult_3 + 11.69)) &
                           (beta_eta1 >= (0.417 * ref_mult_3 - 13.1)) &
                           (beta_eta1 <= (0.526 * ref_mult_3 + 14)))
            p_t, p_g, n_hits_fit, beta, dedx = \
                p_t[event_cuts2], p_g[event_cuts2], n_hits_fit[event_cuts2], \
                beta[event_cuts2], dedx[event_cuts2]
            charge = charge_list(n_hits_fit)

            # Now we flatten everything for use as a pandas dataframe input set.
            p_t, p_g, beta, dedx, charge = \
                ak.to_numpy(ak.flatten(p_t)), ak.to_numpy(ak.flatten(p_g)), \
                ak.to_numpy(ak.flatten(beta)), ak.to_numpy(ak.flatten(dedx)), \
                ak.to_numpy(ak.flatten(charge))

            # Let's add mass.
            p_squared = np.power(p_g, 2)
            b_squared = np.power(beta, 2)
            b_squared[b_squared == 0.0] = 1e-10  # to avoid infinities
            g_squared = (1 - np.power(beta, 2))
            m_squared = np.divide(np.multiply(p_squared, g_squared), b_squared)

            # Last cut for now.
            final_cut = ((p_g <= 3.0) & (m_squared <= 2.0) & (m_squared >= 0))
            p_t, p_g, beta, dedx, charge, m_squared = \
                p_t[final_cut], p_g[final_cut], beta[final_cut], \
                dedx[final_cut], charge[final_cut], m_squared[final_cut]

            # Antiparticle cut. Will they get picked out?
            anti_cut = (charge < 0)
            p_t, p_g, beta, dedx, charge, m_squared = \
                p_t[anti_cut], p_g[anti_cut], beta[anti_cut], \
                dedx[anti_cut], charge[anti_cut], m_squared[anti_cut]

            self.p_t, self.p_g, self.beta, self.dedx, self.charge, self.m_squared = \
                p_t, p_g, beta, dedx, charge, m_squared

            self.data1 = pd.DataFrame(np.vstack((p_t*charge, beta, dedx, p_g*charge, m_squared)).T,
                                      columns=("p_t_q", "beta", "dedx", "p_charge", "mass"))
            self.data2 = pd.DataFrame(np.vstack((dedx, p_g*charge)).T,
                                      columns=("dedx", "p_charge"))
            self.data3 = pd.DataFrame(np.vstack((m_squared, p_g*charge)).T,
                                      columns=("m_squared", "p_charge"))

        except ValueError:  # Usually means an indexing error.
            print("Value error in main data process.")  # Identifies the misbehaving file.
        except KeyError:  # Something is missing in your data.
            print("Key error in main data process.")  # Identifies the misbehaving file.
        except Exception as e:  # For any other issues that might pop up.
            print(e.__class__, "occurred in main data process.")

    def model_kmeans(self, switch=1, n_clusters=10, n_init=10, max_iter=50):
        """Unsupervised, k-means algorithm. This creates hard clusters for
        all tracks, which are found by minimising distances to cluster
        centres. Clusters are "circular" in n-space, so this is not
        ideal. However, it is fast and handles large data sets well.

        Args:
            switch (int): Which data set to fit.
            n_clusters (int): Number of clusters to make. This must be entered
                              manually; set to 10 to include e, p, pi, K (and
                              associated antiparticles), d, and noise.
            n_init (int): Number of initialisations to perform. The algorithm
                          will always keep the best fit as final.
            max_iter (int): Max iterations for a single run through.
        """
        model = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)
        if switch == 1:
            model.fit(self.data1)
            self.kmeans = model.predict(self.data1)
        if switch == 2:
            model.fit(self.data2)
            self.kmeans = model.predict(self.data2)
        if switch == 3:
            model.fit(self.data3)
            self.kmeans = model.predict(self.data3)

    def model_gmm(self, switch=1, n_clusters=10, n_init=10, cov_type='full', max_iter=50):
        """Unsupervised, Gaussian mixture algorithm. Like k-means, this creates
        clusters for all tracks. Unlike k-means, tracks are assigned a
        cluster statistically as the clusters are Gaussian and can overlap.
        Clusters may also be shapes other than circular, so fitting data
        is much more robust. The cost, of course, is computation time due
        to the increased complexity of the algorithm.

        Args:
            switch (int): Which data set to run on.
            n_clusters (int): Number of clusters to make. This must be entered
                              manually; set to 10 to include e, p, pi, K (and
                              associated antiparticles), d, and noise.
            n_init (int): Number of initialisations to perform. The algorithm
                          will always keep the best fit as final.
            max_iter (int): Max iterations for a single run through.
            cov_type (str): Choose from {‘full’, ‘tied’, ‘diag’, ‘spherical’}.
                            From scikit-learn documentation:
                            full = each component has its own general covariance matrix
                            tied = all components share the same general covariance matrix
                            diag = each component has its own diagonal covariance matrix
                            spherical = each component has its own single variance
        """
        model = mixture.GaussianMixture(n_components=n_clusters, covariance_type=cov_type,
                                        n_init=n_init, max_iter=max_iter)
        if switch == 1:
            model.fit(self.data1)
            self.gmm = model.predict(self.data1)
        if switch == 2:
            model.fit(self.data2)
            self.gmm = model.predict(self.data2)
        if switch == 3:
            model.fit(self.data3)
            self.gmm = model.predict(self.data3)
