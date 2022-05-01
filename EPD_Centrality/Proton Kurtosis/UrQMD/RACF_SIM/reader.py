import pandas as pd
import uproot as up
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, Conv1D
from keras.utils.generic_utils import get_custom_objects
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import mixture
from scipy.stats import moment
import time


class Swish(Activation):

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'Swish'


def swish(x, beta=2):
    return x * K.sigmoid(beta * x)


get_custom_objects().update({'Swish': Swish(swish)})


class Mish(Activation):

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'


def mish(x):
    return x * K.tanh(K.softplus(x))


get_custom_objects().update({'Mish': Mish(mish)})


def get_proton_hist(track, pt_low=0.4, pt_mid=0.8, pt_high=2.0,
                    pg_low=1.0, pg_high=3.0, rap_cut=0.5, eta_cut=1.0):
    """This takes the pandas dataframe created in the UrQMD class
    and extrapolates the protons/antiprotons and then stores the
    values in a histogram (proton, antiproton, and net proton
    amounts vs the associated refmult/centrality metric)."""

    # Alternate cuts:
    pg_low = 1000
    pg_high = 1000
    eta_cut = 10.0

    # We need the full index, before cuts.
    full_index = np.unique(track.index.get_level_values(0))

    # First, we select the region of momentum space we want.
    index = ((((track['p_g'] <= pg_low) & (track['p_t'] >= pt_low) & (track['p_t'] <= pt_mid)) |
                ((track['p_g'] <= pg_high) & (track['p_t'] > pt_mid) & (track['p_t'] <= pt_high))) &
                 (abs(track['eta']) <= eta_cut) & (abs(track['rap']) <= rap_cut))
    """index = ((track['p_g'] <= pg_low) & (track['p_t'] >= pt_low) & (track['p_t'] <= pt_mid) &
                 (abs(track['eta']) <= eta_cut) & (abs(track['rap']) <= rap_cut))"""
    track = track[index]

    # Next, we select for protons and antiprotons and derive net protons.
    p_index = (track['pid'] == 1) & (track['q'] == 1)
    ap_index = track['pid'] == -1
    pro = track['pid'][p_index].groupby(level=0).sum().to_numpy()
    apro = abs(track['pid'][ap_index]).groupby(level=0).sum().to_numpy()
    p_index = np.unique(abs(track['pid'][p_index]).index.get_level_values(0))
    ap_index = np.unique(abs(track['pid'][ap_index]).index.get_level_values(0))
    index = np.unique(abs(track['pid'][index]).index.get_level_values(0))
    protons = []
    aprotons = []
    pcount = 0
    apcount = 0
    for i in index:
        if i in p_index:
            protons.append(pro[pcount])
            pcount += 1
        else:
            protons.append(0)
        if i in ap_index:
            aprotons.append(apro[apcount])
            apcount += 1
        else:
            aprotons.append(0)
    protons = np.asarray(protons)
    aprotons = np.asarray(aprotons)
    nprotons = protons - aprotons

    return protons, aprotons, nprotons, index


def get_proton_hist_old(track, pt_low=0.4, pt_high=2.0,
                        rap_cut=0.5):
    """This takes the pandas dataframe created in the UrQMD class
    and extrapolates the protons/antiprotons and then stores the
    values in a histogram (proton, antiproton, and net proton
    amounts vs the associated refmult/centrality metric)."""

    # We need the full index, before cuts.
    full_index = np.unique(track.index.get_level_values(0))

    # First, we select the region of momentum space we want.
    index = (((track['p_t'] >= pt_low) & (track['p_t'] <= pt_high)) &
             (abs(track['rap']) <= rap_cut))
    track = track[index]

    # Next, we select for protons and antiprotons and derive net protons.
    p_index = track['pid'] == 14
    ap_index = track['pid'] == 15
    track.loc[p_index] = 1
    track.loc[ap_index] = 1
    pro = track['pid'][p_index].groupby(level=0).sum().to_numpy()
    apro = abs(track['pid'][ap_index]).groupby(level=0).sum().to_numpy()
    p_index = np.unique(abs(track['pid'][p_index]).index.get_level_values(0))
    ap_index = np.unique(abs(track['pid'][ap_index]).index.get_level_values(0))
    index = np.unique(abs(track['pid'][index]).index.get_level_values(0))
    protons = []
    aprotons = []
    pcount = 0
    apcount = 0
    for i in index:
        if i in p_index:
            protons.append(pro[pcount])
            pcount += 1
        else:
            protons.append(0)
        if i in ap_index:
            aprotons.append(apro[apcount])
            apcount += 1
        else:
            aprotons.append(0)
    protons = np.asarray(protons)
    aprotons = np.asarray(aprotons)
    nprotons = protons - aprotons

    return protons, aprotons, nprotons, index


def get_proton_hist_old_2(track, pt_low=0.4, pt_high=2.0,
                          rap_cut=0.5):
    """This takes the pandas dataframe created in the UrQMD class
    and extrapolates the protons/antiprotons and then stores the
    values in a histogram (proton, antiproton, and net proton
    amounts vs the associated refmult/centrality metric)."""

    # We need the full index, before cuts.
    full_index = np.unique(track.index.get_level_values(0))

    # First, we select the region of momentum space we want.
    index = (((track['p_t'] >= pt_low) & (track['p_t'] <= pt_high)) &
             (abs(track['rap']) <= rap_cut))
    track = track[index]

    # Next, we select for protons and antiprotons and derive net protons.
    p_index = track['pid'] == 2
    ap_index = track['pid'] == 5
    track.loc[p_index] = 1
    track.loc[ap_index] = 1
    pro = track['pid'][p_index].groupby(level=0).sum().to_numpy()
    apro = abs(track['pid'][ap_index]).groupby(level=0).sum().to_numpy()
    p_index = np.unique(abs(track['pid'][p_index]).index.get_level_values(0))
    ap_index = np.unique(abs(track['pid'][ap_index]).index.get_level_values(0))
    index = np.unique(abs(track['pid'][index]).index.get_level_values(0))
    protons = []
    aprotons = []
    pcount = 0
    apcount = 0
    for i in index:
        if i in p_index:
            protons.append(pro[pcount])
            pcount += 1
        else:
            protons.append(0)
        if i in ap_index:
            aprotons.append(apro[apcount])
            apcount += 1
        else:
            aprotons.append(0)
    protons = np.asarray(protons)
    aprotons = np.asarray(aprotons)
    nprotons = protons - aprotons

    return protons, aprotons, nprotons, index


class UrQMD:

    def __init__(self):
        """These are the available variables from the UrQMD sim."""
        self.refmult = None
        self.refmult_full = None
        self.rings = None
        self.b = None
        self.x = None
        self.y = None
        self.z = None
        self.pid = None
        self.p_x = None
        self.p_y = None
        self.p_z = None
        self.p_t = None
        self.p_g = None
        self.e = None
        self.eta = None
        self.q = None
        self.rap = None
        self.m = None
        self.pr_id = None
        self.event = None
        self.track = None
        self.epd_eta_ranges = np.array(((5.09, 4.42), (4.42, 4.03), (4.03, 3.74), (3.74, 3.47),
                                        (3.47, 3.26), (3.26, 3.08), (3.08, 2.94), (2.94, 2.81),
                                        (2.81, 2.69), (2.69, 2.59), (2.59, 2.50), (2.50, 2.41),
                                        (2.41, 2.34), (2.34, 2.27), (2.27, 2.20), (2.20, 2.14),
                                        (-5.09, -4.42), (-4.42, -4.03), (-4.03, -3.74), (-3.74, -3.47),
                                        (-3.47, -3.26), (-3.26, -3.08), (-3.08, -2.94), (-2.94, -2.81),
                                        (-2.81, -2.69), (-2.69, -2.59), (-2.59, -2.50), (-2.50, -2.41),
                                        (-2.41, -2.34), (-2.34, -2.27), (-2.27, -2.20), (-2.20, -2.14)))

    def import_data_arr(self, data, old=False):
        """This imports the data as arrays (numpy for event level
        quantities and awkward for track level quantities). You
        must have the latest versions of uproot and awkward installed
        on your machine (uproot4 and awkward 1.0 as of the time of
        this writing).
        If you don't, use: pip install uproot awkward.
        Args:
            data (str): The path to the UrQMD ROOT file
            old (bool): Set to True if using the older UrQMD data sets"""
        if old is False:
            try:
                data = up.open(data)["UrQmdtree"]
                self.b = data["Impact"].array(library='np')
                """
                # Everything's off for now but b just to get event number.
                self.x = data["X"].array(library='np')
                self.y = data["Y"].array(library='np')
                self.z = data["Z"].array(library='np')
                self.pid = data["PID"].array()
                self.p_x = data["Px"].array()
                self.p_y = data["Py"].array()
                self.p_z = data["Pz"].array()
                self.p_t = data["Pt"].array()
                self.p_g = data["E"].array()
                self.e = data["E"].array()
                self.eta = data["Eta"].array()
                self.q = data["Charge"].array()
                self.rap = data["RAP"].array()
                self.m = data["M"].array()
                self.pr_id = data["PRid"].array()
                """

                """UrQMD IDs as follows:
                p / n = 1, aparticles are - 1
                pi = 101
                k = 106
                Further particle differentiation can be found with charge sign."""
                """
                self.refmult = ak.sum(ak.where((self.pid == 101) | (self.pid == 106) & (self.q != 0)
                                               & (abs(self.eta) <= 1.0), 1, 0), axis=-1)
                self.refmult_full = ak.sum(ak.where((self.pid == 101) | (self.pid == 106) | (self.pid == 1)
                                                    | (self.pid == -1) & (self.q != 0)
                                                    & (abs(self.eta) <= 1.0), 1, 0), axis=-1)
                self.refmult = ak.to_numpy(self.refmult)
                self.refmult_full = ak.to_numpy(self.refmult_full)
                e_data = {'b': self.b, 'refmult': self.refmult, 'refmult_full': self.refmult_full}
                rings = []
                for i in range(32):
                    rings.append([])
                    if i < 16:
                        rings[i] = ak.sum(ak.where(((self.pid == 101) | (self.pid == 106) | (self.pid == 1)
                                                    | (self.pid == -1)) & (self.q != 0) &
                                                   (self.eta <= self.epd_eta_ranges[i][0])
                                                   & (self.eta > self.epd_eta_ranges[i][1]), 1, 0), axis=-1)
                    else:
                        rings[i] = ak.sum(ak.where(((self.pid == 101) | (self.pid == 106) | (self.pid == 1)
                                                    | (self.pid == -1)) & (self.q != 0) &
                                                   (self.eta >= self.epd_eta_ranges[i][0])
                                                   & (self.eta < self.epd_eta_ranges[i][1]), 1, 0), axis=-1)
                self.rings = np.asarray(rings)
                """
            except ValueError:  # Skip empty picos.
                print("ValueError at: " + data)  # Identifies the misbehaving file.
            except KeyError:  # Skip non-empty picos that have no data.
                print("KeyError at: " + data)  # Identifies the misbehaving file.
        else:
            try:
                data = up.open(data)
                if 'h1;1' in data.keys():
                    data = data['h1']
                    self.b = data["parimp"].array(library='np')
                else:
                    data = data['urqmd']
                    self.b = data["b"].array(library='np')
                """
                # Everything's off for now but b just to get event number.
                self.x = data["X"].array(library='np')
                self.y = data["Y"].array(library='np')
                self.z = data["Z"].array(library='np')
                self.pid = data["PID"].array()
                self.p_x = data["Px"].array()
                self.p_y = data["Py"].array()
                self.p_z = data["Pz"].array()
                self.p_t = data["Pt"].array()
                self.p_g = data["E"].array()
                self.e = data["E"].array()
                self.eta = data["Eta"].array()
                self.q = data["Charge"].array()
                self.rap = data["RAP"].array()
                self.m = data["M"].array()
                self.pr_id = data["PRid"].array()
                self.refmult = ak.sum(ak.where((self.pid == 101) | (self.pid == 106) & (self.q != 0)
                                               & (abs(self.eta) <= 1.0), 1, 0), axis=-1)
                self.refmult_full = ak.sum(ak.where((self.pid == 101) | (self.pid == 106) | (self.pid == 1)
                                                    | (self.pid == -1) & (self.q != 0)
                                                    & (abs(self.eta) <= 1.0), 1, 0), axis=-1)
                self.refmult = ak.to_numpy(self.refmult)
                self.refmult_full = ak.to_numpy(self.refmult_full)
                e_data = {'b': self.b, 'refmult': self.refmult, 'refmult_full': self.refmult_full}
                rings = []
                for i in range(32):
                    rings.append([])
                    if i < 16:
                        rings[i] = ak.sum(ak.where(((self.pid == 101) | (self.pid == 106) | (self.pid == 1)
                                                    | (self.pid == -1)) & (self.q != 0) &
                                                   (self.eta <= self.epd_eta_ranges[i][0])
                                                   & (self.eta > self.epd_eta_ranges[i][1]), 1, 0), axis=-1)
                    else:
                        rings[i] = ak.sum(ak.where(((self.pid == 101) | (self.pid == 106) | (self.pid == 1)
                                                    | (self.pid == -1)) & (self.q != 0) &
                                                   (self.eta >= self.epd_eta_ranges[i][0])
                                                   & (self.eta < self.epd_eta_ranges[i][1]), 1, 0), axis=-1)
                self.rings = np.asarray(rings)
                """
            except ValueError:  # Skip empty picos.
                print("ValueError at: " + data)  # Identifies the misbehaving file.
            except KeyError:  # Skip non-empty picos that have no data.
                print("KeyError at: " + data)  # Identifies the misbehaving file.

    def import_data_df(self, m1, m2, m3, m4, data, ML=True):
        """This imports the data as pandas dataframes, one for event level
        quantities and one for track level quantities. You must have the
        latest versions of uproot and awkward installed on your machine
        (uproot4 and awkward 1.0 as of the time of this writing).
        If you don't, use: pip install uproot awkward.
        Args:
            data (str): The path to the UrQMD ROOT file
            ML (bool): True if using ML fits
            m1 (model): LW ML model
            m2 (model): ReLU ML model
            m3 (model: Swish ML model
            m4 (model): CNN ML model (ReLU)"""
        try:
            with(up.open(data)["UrQmdtree"]) as data:
                time0 = time.time()
                # Note: A lot of these are turned off to save processing time.
                self.b = data["Impact"].array(library='np')
                # self.x = data["X"].array(library='np')
                # self.y = data["Y"].array(library='np')
                # self.z = data["Z"].array(library='np')
                self.pid = data["PID"].array()
                # self.p_x = data["Px"].array()
                # self.p_y = data["Py"].array()
                # self.p_z = data["Pz"].array()
                self.p_t = data["Pt"].array()
                self.p_g = data["E"].array()
                # self.e = data["E"].array()
                self.eta = data["Eta"].array()
                self.q = data["Charge"].array()
                self.rap = data["RAP"].array()
                # self.m = data["M"].array()
                # self.pr_id = data["PRid"].array()
                time1 = time.time()
                # print("Initialisation time:", time1-time0)
                """UrQMD IDs as follows:
                p / n = 1, aparticles are - 1
                pi = 101
                k = 106
                Further particle differentiation can be found with charge sign."""
                self.refmult = ak.sum(ak.where(((self.pid == 101) | (self.pid == 106)) & (self.q != 0)
                                               & (abs(self.eta) <= 1.0), 1, 0), axis=-1)
                self.refmult_full = ak.sum(ak.where(((self.pid == 101) | (self.pid == 106) | (self.pid == 1)
                                                    | (self.pid == -1)) & (self.q != 0)
                                                    & (abs(self.eta) <= 0.5), 1, 0), axis=-1)
                self.refmult = ak.to_numpy(self.refmult)
                self.refmult_full = ak.to_numpy(self.refmult_full)
                e_data = {'b': self.b, 'refmult': self.refmult, 'refmult_full': self.refmult_full}
                self.event = pd.DataFrame(e_data)
                time2 = time.time()
                # print("Event generation time:", time2-time1)
                rings = []
                for i in range(32):
                    rings.append([])
                    if i < 16:
                        rings[i] = ak.sum(ak.where(((self.pid == 101) | (self.pid == 106) | (self.pid == 1)
                                                    | (self.pid == -1)) & (self.q != 0) &
                                                   (self.eta <= self.epd_eta_ranges[i][0])
                                                   & (self.eta > self.epd_eta_ranges[i][1]), 1, 0), axis=-1)
                    else:
                        rings[i] = ak.sum(ak.where(((self.pid == 101) | (self.pid == 106) | (self.pid == 1)
                                                    | (self.pid == -1)) & (self.q != 0) &
                                                   (self.eta >= self.epd_eta_ranges[i][0])
                                                   & (self.eta < self.epd_eta_ranges[i][1]), 1, 0), axis=-1)
                    self.event['ring{}'.format(i + 1)] = rings[i]
                self.rings = np.asarray(rings)
                time3 = time.time()
                # print("Ring generation time:", time3-time2)

                if ML is True:
                    ring_entries = np.add(self.rings[:16], self.rings[16:]).T
                    self.event['epd_linear'] = m1.predict(ring_entries, verbose=0).T.flatten()
                    self.event['epd_relu'] = m2.predict(ring_entries, verbose=0).T.flatten()
                    self.event['epd_swish'] = m3.predict(ring_entries, verbose=0).T.flatten()
                    self.event['epd_CNN'] = m4.predict(np.expand_dims(ring_entries, axis=2),
                                                       verbose=0).T.flatten()
                time4 = time.time()
                # print("ML generation time:", time4-time3)
                # Make the pandas track dataframe.
                # Note: Some things are turned off to save processing time.
                self.track = pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["event", "track"]))
                levels = ["event", "track"]
                pid = ak.to_pandas(data["PID"].array(), levelname=lambda j: levels[j], anonymous='pid')
                self.track = pd.concat((self.track, pid))
                # self.track['x'] = ak.to_pandas(data["X"].array(), levelname=lambda j: levels[j], anonymous='x')
                # self.track['y'] = ak.to_pandas(data["Y"].array(), levelname=lambda j: levels[j], anonymous='y')
                # self.track["z"] = ak.to_pandas(data["Z"].array(), levelname=lambda j: levels[j], anonymous='z')
                # self.track["p_x"] = ak.to_pandas(data["Px"].array(), levelname=lambda j: levels[j], anonymous='p_x')
                # self.track["p_y"] = ak.to_pandas(data["Py"].array(), levelname=lambda j: levels[j], anonymous='p_y')
                # self.track["p_z"] = ak.to_pandas(data["Pz"].array(), levelname=lambda j: levels[j], anonymous='p_z')
                self.track["p_t"] = ak.to_pandas(data["Pt"].array(), levelname=lambda j: levels[j], anonymous='p_t')
                self.track["p_g"] = ak.to_pandas(data["E"].array(), levelname=lambda j: levels[j], anonymous='p_g')
                self.track["eta"] = ak.to_pandas(data["Eta"].array(), levelname=lambda j: levels[j], anonymous='eta')
                self.track["q"] = ak.to_pandas(data["Charge"].array(), levelname=lambda j: levels[j], anonymous='q')
                self.track["rap"] = ak.to_pandas(data["RAP"].array(), levelname=lambda j: levels[j], anonymous='rap')
                # self.track["m"] = ak.to_pandas(data["M"].array(), levelname=lambda j: levels[j], anonymous='m')
                time5 = time.time()
                # print("Track generation time:", time5-time4)

        except ValueError:  # Skip empty picos.
            print("ValueError at: " + data)  # Identifies the misbehaving file.
        except KeyError:  # Skip non-empty picos that have no data.
            print("KeyError at: " + data)  # Identifies the misbehaving file.

    def import_data_df_old(self, m1, m2, m3, m4, data, ML=True):
        """This imports the data as pandas dataframes, one for event level
        quantities and one for track level quantities. You must have the
        latest versions of uproot and awkward installed on your machine
        (uproot4 and awkward 1.0 as of the time of this writing).
        If you don't, use: pip install uproot awkward.
        Args:
            data (str): The path to the UrQMD ROOT file
            ML (bool): True if using ML fits
            m1 (model): LW ML model
            m2 (model): ReLU ML model
            m3 (model: Swish ML model
            m4 (model): CNN ML model (ReLU)"""
        try:
            with(up.open(data)["h1"]) as data:
                self.b = data["parimp"].array(library='np')
                self.pid = data["igid"].array()
                self.p_x = data["gpx"].array()
                self.p_y = data["gpy"].array()
                self.p_z = data["gpz"].array()
                self.p_t = np.sqrt(self.p_x**2 + self.p_y**2)
                self.p_g = np.sqrt(self.p_x**2 + self.p_y**2 + self.p_z**2)
                with np.errstate(divide='ignore', invalid='ignore'):
                    eta = (1/2)*np.log((self.p_g+self.p_z)/(self.p_g-self.p_z))
                count_dims = ak.num(eta)
                eta = ak.flatten(eta)
                eta = np.where(eta == np.inf, 10, eta)
                eta = np.where(eta == -np.inf, -10, eta)
                self.eta = ak.unflatten(eta, count_dims)
                p_mass = 0.938272
                pi_mass = 0.13957
                k_mass = 0.497648
                count_dims = ak.num(self.pid)
                m = ak.flatten(self.pid)
                m = np.where((m == 8) | (m == 9), pi_mass, m)
                m = np.where((m == 11) | (m == 12), k_mass, m)
                m = np.where((m == 14) | (m == 15), p_mass, m)
                self.m = ak.unflatten(m, count_dims)
                self.e = np.sqrt(self.m**2 + self.p_g**2)
                self.rap = (1/2)*np.log((self.e+self.p_z)/(self.e-self.p_z))
                """PID for this data set are as follows:
                8=pi+, 9=pi-, 11=k+, 12=k-, 14=proton, 15=pbar
                """
                self.refmult = ak.sum(ak.where(((self.pid == 8) | (self.pid == 9) |
                                                (self.pid == 10) | (self.pid == 11))
                                               & (abs(self.eta) <= 1.0), 1, 0), axis=-1)
                self.refmult_full = ak.sum(ak.where(((self.pid == 8) | (self.pid == 9) |
                                                     (self.pid == 10) | (self.pid == 11) |
                                                     (self.pid == 14) | (self.pid == 15))
                                                    & (abs(self.eta) <= 0.5), 1, 0), axis=-1)
                self.refmult = ak.to_numpy(self.refmult)
                self.refmult_full = ak.to_numpy(self.refmult_full)
                e_data = {'b': self.b, 'refmult': self.refmult, 'refmult_full': self.refmult_full}
                self.event = pd.DataFrame(e_data)
                rings = []
                for i in range(32):
                    rings.append([])
                    if i < 16:
                        rings[i] = ak.sum(ak.where(((self.pid == 8) | (self.pid == 9) | (self.pid == 11)
                                                    | (self.pid == 12) | (self.pid == 14) | (self.pid == 15)) &
                                                   (self.eta <= self.epd_eta_ranges[i][0])
                                                   & (self.eta > self.epd_eta_ranges[i][1]), 1, 0), axis=-1)
                    else:
                        rings[i] = ak.sum(ak.where(((self.pid == 8) | (self.pid == 9) | (self.pid == 11)
                                                    | (self.pid == 12) | (self.pid == 14) | (self.pid == 15)) &
                                                   (self.eta >= self.epd_eta_ranges[i][0])
                                                   & (self.eta < self.epd_eta_ranges[i][1]), 1, 0), axis=-1)
                    self.event['ring{}'.format(i + 1)] = rings[i]
                self.rings = np.asarray(rings)

                # TODO Clean this whole mess up (but for now it's functional).
                if ML is True:
                    ring_entries = np.add(self.rings[:16], self.rings[16:]).T
                    self.event['epd_linear'] = m1.predict(ring_entries, verbose=0).T.flatten()
                    self.event['epd_relu'] = m2.predict(ring_entries, verbose=0).T.flatten()
                    self.event['epd_swish'] = m3.predict(ring_entries, verbose=0).T.flatten()
                    self.event['epd_CNN'] = m4.predict(np.expand_dims(ring_entries, axis=2),
                                                       verbose=0).T.flatten()
                # Make the pandas track dataframe.
                self.track = pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["event", "track"]))
                levels = ["event", "track"]
                pid = ak.to_pandas(self.pid, levelname=lambda j: levels[j], anonymous='pid')
                self.track = pd.concat((self.track, pid))
                self.track["p_x"] = ak.to_pandas(self.p_x, levelname=lambda j: levels[j], anonymous='p_x')
                self.track["p_y"] = ak.to_pandas(self.p_y, levelname=lambda j: levels[j], anonymous='p_y')
                self.track["p_z"] = ak.to_pandas(self.p_z, levelname=lambda j: levels[j], anonymous='p_z')
                self.track["p_t"] = ak.to_pandas(self.p_t, levelname=lambda j: levels[j], anonymous='p_t')
                self.track["p_g"] = ak.to_pandas(self.p_g, levelname=lambda j: levels[j], anonymous='p_g')
                self.track["eta"] = ak.to_pandas(self.eta, levelname=lambda j: levels[j], anonymous='eta')
                self.track["rap"] = ak.to_pandas(self.rap, levelname=lambda j: levels[j], anonymous='rap')
                self.track["m"] = ak.to_pandas(self.m, levelname=lambda j: levels[j], anonymous='m')

        except ValueError:  # Skip empty picos.
            print("ValueError at: " + data)  # Identifies the misbehaving file.
        except KeyError:  # Skip non-empty picos that have no data.
            print("KeyError at: " + data)  # Identifies the misbehaving file.

    def import_data_df_old_2(self, m1, m2, m3, m4, data, ML=True):
        """This imports the data as pandas dataframes, one for event level
        quantities and one for track level quantities. You must have the
        latest versions of uproot and awkward installed on your machine
        (uproot4 and awkward 1.0 as of the time of this writing).
        If you don't, use: pip install uproot awkward.
        Args:
            data (str): The path to the UrQMD ROOT file
            ML (bool): True if using ML fits
            m1 (model): LW ML model
            m2 (model): ReLU ML model
            m3 (model: Swish ML model
            m4 (model): CNN ML model (ReLU)"""
        try:
            with(up.open(data)["urqmd"]) as data:
                self.b = data["b"].array(library='np')
                self.pid = data["pid"].array()
                self.p_t = data['ptbin'].array()/100.0
                self.eta = data['etabin'].array()*0.02
                p_mass = 0.938272
                pi_mass = 0.13957
                k_mass = 0.497648
                count_dims = ak.num(self.pid)
                m = ak.flatten(self.pid)
                """PID for this data set are as follows:
                0=pi+ 1=k+ 2=p 3=pi- 4=k- 5=pbar 6=pi0 7=eta 8=k0 9=n 10=nbar 11=default
                """
                m = np.where((m == 0) | (m == 3), pi_mass, m)
                m = np.where((m == 1) | (m == 4), k_mass, m)
                m = np.where((m == 2) | (m == 5), p_mass, m)
                pt = ak.flatten(self.p_t)
                eta = ak.flatten(self.eta)
                numerator = np.sqrt((pt**2 * np.cosh(eta)**2) + m**2) + pt*np.sinh(eta)
                denominator = np.sqrt((pt**2 * np.cosh(eta)**2) + m**2) - pt*np.sinh(eta)
                self.m = ak.unflatten(m, count_dims)
                rap = (1 / 2) * (np.log(numerator) - np.log(denominator))
                self.rap = ak.unflatten(rap, count_dims)
                self.refmult = ak.sum(ak.where(((self.pid == 0) | (self.pid == 1) |
                                                (self.pid == 3) | (self.pid == 4))
                                               & (abs(self.eta) <= 1.0), 1, 0), axis=-1)
                self.refmult_full = ak.sum(ak.where(((self.pid == 0) | (self.pid == 1) |
                                                     (self.pid == 2) | (self.pid == 3) |
                                                     (self.pid == 4) | (self.pid == 5))
                                                    & (abs(self.eta) <= 0.5), 1, 0), axis=-1)
                self.refmult = ak.to_numpy(self.refmult)
                self.refmult_full = ak.to_numpy(self.refmult_full)
                e_data = {'b': self.b, 'refmult': self.refmult, 'refmult_full': self.refmult_full}
                self.event = pd.DataFrame(e_data)
                rings = []
                for i in range(32):
                    rings.append([])
                    if i < 16:
                        rings[i] = ak.sum(ak.where(((self.pid == 0) | (self.pid == 1) |
                                                    (self.pid == 2) | (self.pid == 3) |
                                                    (self.pid == 4) | (self.pid == 5)) &
                                                   (self.eta <= self.epd_eta_ranges[i][0])
                                                   & (self.eta > self.epd_eta_ranges[i][1]), 1, 0), axis=-1)
                    else:
                        rings[i] = ak.sum(ak.where(((self.pid == 0) | (self.pid == 1) |
                                                    (self.pid == 2) | (self.pid == 3) |
                                                    (self.pid == 4) | (self.pid == 5)) &
                                                   (self.eta >= self.epd_eta_ranges[i][0])
                                                   & (self.eta < self.epd_eta_ranges[i][1]), 1, 0), axis=-1)
                    self.event['ring{}'.format(i + 1)] = rings[i]
                self.rings = np.asarray(rings)

                # TODO Clean this whole mess up (but for now it's functional).
                if ML is True:
                    ring_entries = np.add(self.rings[:16], self.rings[16:]).T
                    self.event['epd_linear'] = m1.predict(ring_entries, verbose=0).T.flatten()
                    self.event['epd_relu'] = m2.predict(ring_entries, verbose=0).T.flatten()
                    self.event['epd_swish'] = m3.predict(ring_entries, verbose=0).T.flatten()
                    self.event['epd_CNN'] = m4.predict(np.expand_dims(ring_entries, axis=2),
                                                       verbose=0).T.flatten()
                # Make the pandas track dataframe.
                self.track = pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["event", "track"]))
                levels = ["event", "track"]
                pid = ak.to_pandas(self.pid, levelname=lambda j: levels[j], anonymous='pid')
                self.track = pd.concat((self.track, pid))
                self.track["p_t"] = ak.to_pandas(self.p_t, levelname=lambda j: levels[j], anonymous='p_t')
                self.track["eta"] = ak.to_pandas(self.eta, levelname=lambda j: levels[j], anonymous='eta')
                self.track["rap"] = ak.to_pandas(self.rap, levelname=lambda j: levels[j], anonymous='rap')
                self.track["m"] = ak.to_pandas(self.m, levelname=lambda j: levels[j], anonymous='m')

        except ValueError:  # Skip empty picos.
            print("ValueError at: " + data)  # Identifies the misbehaving file.
        except KeyError:  # Skip non-empty picos that have no data.
            print("KeyError at: " + data)  # Identifies the misbehaving file.


def u_n(arr1, arr2, u1, power=2):
    """
    This code will find the moment (n>= 2) for a 2D histogram. The return is the
    array of moments for the given reference multiplicity.
        Args:
            arr1: The MxN matrix of the correlation amounts from the 2D histogram.
            arr2: The XxY axis from the 2D hostogram.
            u1: The array of the means.
            power: The power of the moment you're looking for.
    """
    n = np.sum(arr1, axis=1)
    n[n == 0] = 1e-10  # To avoid infinities with blank results.
    dN = np.repeat(np.expand_dims(arr2[1][:-1], axis=0), len(arr1), axis=0)
    for i in range(len(dN)):
        dN[i] -= u1[i]
    dN = arr1 * np.power(dN, power)
    with np.errstate(divide='ignore', invalid='ignore'):
        un = np.where(n > 0, np.divide(np.sum(dN, axis=1), n), 0)
    return np.asarray(un)


def u_1(arr1, arr2):
    """
    This is for finding the mean (u_1) of a 2D histogram. The return is the
    array of means for the given reference multiplicity.
        Args:
            arr1: The MxN matrix of the correlation amounts from the 2D histogram.
            arr2: The XxY axis from the 2D hostogram.
    """
    n = np.sum(arr1, axis=1)
    n[n == 0] = 1e-6
    n = np.asarray(n).astype(np.float)
    totals = np.sum(arr1 * arr2[1][:-1], axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        totals = np.asarray(totals).astype(np.float)
        u1 = np.where(n >= 1e-5, totals/n, 0)
    return np.asarray(u1)


def err(arr1, arr2, u1, power=1):
    """
    This is for finding the error in the moments of a 2D histogram. The return is the
    array of moment errors for the given reference multiplicity and power.
    Error is found using Delta Theorem.
        Args:
            arr1: The MxN matrix of the correlation amounts from the 2D histogram.
            arr2: The XxY axis from the 2D hostogram.
            u1: The array of means.
            power: Power of the error being requested.
    """
    n = np.sum(arr1, axis=1)
    n[n == 0] = 1e-10  # To avoid infinities with blank results.
    e = np.zeros(len(n))
    with np.errstate(divide='ignore', invalid='ignore'):
        if power == 1:
            u = u_n(arr1, arr2, u1, 2)
            u[n == 1e-10] = 0
            u = np.divide(u, n)
            u[u == 0] = 1e-6
            u = np.asarray(u).astype(np.float)
            e = np.where(u >= 1e-5, np.sqrt(u), 0)
        if power == 2:
            u = (-u_n(arr1, arr2, u1, 2)) ** 2 + u_n(arr1, arr2, u1, 2)
            u[n == 1e-10] = 0
            u = np.divide(u, n)
            u[u == 0] = 1e-6
            u = np.asarray(u).astype(np.float)
            e = np.where(u >= 1e-5, np.sqrt(u), 0)
        if power == 3:
            u2 = u_n(arr1, arr2, u1, 2)
            u3 = u_n(arr1, arr2, u1, 3)
            u4 = u_n(arr1, arr2, u1, 4)
            u6 = u_n(arr1, arr2, u1, 6)
            u = 9 * (u2 ** 3) - 6 * u2 * u4 - u3 ** 2 + u6
            u[n == 1e-10] = 0
            u = np.divide(u, n)
            u[u == 0] = 1e-6
            u = np.asarray(u).astype(np.float)
            e = np.where(u >= 1e-5, np.sqrt(u), 0)
        if power == 4:
            u2 = u_n(arr1, arr2, u1, 2)
            u3 = u_n(arr1, arr2, u1, 3)
            u4 = u_n(arr1, arr2, u1, 4)
            u5 = u_n(arr1, arr2, u1, 5)
            u6 = u_n(arr1, arr2, u1, 6)
            u8 = u_n(arr1, arr2, u1, 8)
            u = -36 * (u2 ** 4) + 48 * (u2 ** 2) * u4 + 64 * (u3 ** 2) * u2 - 12 * u2 * u6 - 8 * u3 * u5 \
                - u4 ** 2 + u8
            u[n == 1e-10] = 0
            u = np.divide(u, n)
            u[u == 0] = 1e-6
            u = np.asarray(u).astype(np.float)
            e = np.where(u >= 1e-5, np.sqrt(u), 0)
    if np.sum(e) == 0:
        print("Power =", power, "and e == 0, no error found.")
    return e


def err_rat_(arr1, arr2, u1, power=3):
    """
    This is for finding the error in the ratio values using delta theorem.
        Args:
            arr1: The MxN matrix of the correlation amounts from the 2D histogram.
            arr2: The XxY axis from the 2D hostogram.
            u1: The array of means.
            power: Power of the error being requested.
    """
    n = np.sum(arr1, axis=1)
    n[n == 0] = 1e-6  # To avoid infinities with blank results.
    n = np.asarray(n)
    e = np.zeros(len(n))
    u_2 = u_n(arr1, arr2, u1, 2)
    u_3 = u_n(arr1, arr2, u1, 3)
    u_4 = u_n(arr1, arr2, u1, 4)
    u_5 = u_n(arr1, arr2, u1, 5)
    u_6 = u_n(arr1, arr2, u1, 6)
    u_8 = u_n(arr1, arr2, u1, 8)
    u_2[u_2 == 0] = 1e-6
    u_3[u_3 == 0] = 1e-6
    u_4[u_4 == 0] = 1e-6
    u_5[u_5 == 0] = 1e-6
    u_6[u_6 == 0] = 1e-6
    u_8[u_8 == 0] = 1e-6
    with np.errstate(divide='ignore', invalid='ignore'):
        if power == 1:
            u = u_n(arr1, arr2, u1, 2)
            u = np.divide(u, n)
            u[u == 0] = 1e-6
            u = np.asarray(u).astype(np.float)
            e = np.where(u >= 1e-5, np.sqrt(u), 0)
        if power == 2:
            arr1c = np.copy(arr1)
            arr1c[arr1c == 0] = np.nan
            # n_mean = np.nanmean(arr1c.astype(np.float), axis=1)
            n_mean = u1
            n_mean[np.isnan(n_mean)] = 1e-6
            u = np.where(n_mean > 1e-5, np.divide(np.subtract(u_4, np.power(u_2, 2)), np.power(n_mean, 2)), 0) - \
                np.where(n_mean > 1e-5, np.divide(2*np.multiply(u_2, u_3), np.power(n_mean, 3)), 0) + \
                np.where(n_mean > 1e-5, np.divide(np.power(u_2, 3), np.power(n_mean, 4)), 0)
            u[u == 0] = 1e-6
            u = np.asarray(u).astype(np.float)
            e = np.where(u >= 1e-5, np.sqrt(u), 0)
        if power == 3:
            u = 9*np.power(u_2, 2) - \
                np.where(u_2 > 0, np.divide(6*u_4, u_2), 0) + \
                np.where(u_2 > 0, np.divide(np.add(6*u_3, u_6), np.power(u_2, 2)), 0) - \
                np.where(u_2 > 0, np.divide(np.multiply(2*u_3, u_5), np.power(u_2, 3)), 0) + \
                np.where(u_2 > 0, np.divide(np.multiply(np.power(u_3, 2), u_4), np.power(u_2, 4)), 0)
            u[u == 0] = 1e-6
            u = np.asarray(u).astype(np.float)
            e = np.where(u >= 1e-5, np.sqrt(u), 0)
        if power == 4:
            u = -9*np.power(u_2, 2) + 9*u_4 + \
                np.where(u_2 > 0, np.divide(np.subtract(40*np.power(u_3, 2), 6*u_6), u_2), 0) + \
                np.where(u_2 > 0, np.divide(np.subtract(np.add(u_8, 6*np.power(u_4, 2)),
                                                        8*np.multiply(u_3, u_5)),
                                            np.power(u_2, 2)), 0) + \
                np.where(u_2 > 0, np.divide(np.subtract(8*np.multiply(np.power(u_3, 4), u_4),
                                                        2*np.multiply(u_4, u_6)),
                                            np.power(u_2, 3)), 0) + \
                np.where(u_2 > 0, np.divide(np.power(u_4, 3), np.power(u_2, 4)), 0)
            u[u == 0] = 1e-6
            u = np.asarray(u).astype(np.float)
            e = np.where(u >= 1e-5, np.sqrt(u), 0)
    return e


def err_rat(arr1, arr2, u1, power=3):
    """
    This is for finding the error in the ratio values using delta theorem.
        Args:
            arr1: The MxN matrix of the correlation amounts from the 2D histogram.
            arr2: The XxY axis from the 2D hostogram.
            u1: The array of means.
            power: Power of the error being requested.
    """
    n = np.sum(arr1, axis=1)
    n[n == 0] = 1e-6  # To avoid infinities with blank results.
    n = np.asarray(n)
    e = np.zeros(len(n))
    u_2 = u_n(arr1, arr2, u1, 2)
    u_3 = u_n(arr1, arr2, u1, 3)
    u_4 = u_n(arr1, arr2, u1, 4)
    u_5 = u_n(arr1, arr2, u1, 5)
    u_6 = u_n(arr1, arr2, u1, 6)
    u_8 = u_n(arr1, arr2, u1, 8)
    u_2[u_2 == 0] = 1e-6
    u_3[u_3 == 0] = 1e-6
    u_4[u_4 == 0] = 1e-6
    u_5[u_5 == 0] = 1e-6
    u_6[u_6 == 0] = 1e-6
    u_8[u_8 == 0] = 1e-6
    with np.errstate(divide='ignore', invalid='ignore'):
        if power == 1:
            u = u_n(arr1, arr2, u1, 2)
            u = np.divide(u, n)
            u[u == 0] = 1e-6
            u = np.asarray(u).astype(np.float)
            e = np.where(u >= 1e-5, np.sqrt(u), 0)
        if power == 2:
            arr1c = np.copy(arr1)
            arr1c[arr1c == 0] = np.nan
            n_mean = np.nanmean(arr1c.astype(np.float), axis=1)
            n_mean[np.isnan(n_mean)] = 1e-6
            n_mean = u1
            u = np.where(n_mean > 1e-5, np.divide(np.subtract(u_4, np.power(u_2, 2)), np.power(n_mean, 2)), 0) - \
                np.where(n_mean > 1e-5, np.divide(2*np.multiply(u_2, u_3), np.power(n_mean, 3)), 0) + \
                np.where(n_mean > 1e-5, np.divide(np.power(u_2, 3), np.power(n_mean, 4)), 0)
            u[u == 0] = 1e-6
            u = np.asarray(u).astype(np.float)
            u = np.divide(u, n)
            e = np.where(u >= 1e-5, np.sqrt(u), 0)
        if power == 3:
            u = 9*np.power(u_2, 2) - \
                np.where(u_2 > 0, np.divide(6*u_4, u_2), 0) + \
                np.where(u_2 > 0, np.divide(np.add(6*u_3, u_6), np.power(u_2, 2)), 0) - \
                np.where(u_2 > 0, np.divide(np.multiply(2*u_3, u_5), np.power(u_2, 3)), 0) + \
                np.where(u_2 > 0, np.divide(np.multiply(np.power(u_3, 2), u_4), np.power(u_2, 4)), 0)
            u[u == 0] = 1e-6
            u = np.asarray(u).astype(np.float)
            u = np.divide(u, n)
            e = np.where(u >= 1e-5, np.sqrt(u), 0)
        if power == 4:
            u = -9*np.power(u_2, 2) + 9*u_4 + \
                np.where(u_2 > 0, np.divide(np.subtract(40*np.power(u_3, 2), 6*u_6), u_2), 0) + \
                np.where(u_2 > 0, np.divide(np.subtract(np.add(u_8, 6*np.power(u_4, 2)),
                                                        8*np.multiply(u_3, u_5)),
                                            np.power(u_2, 2)), 0) + \
                np.where(u_2 > 0, np.divide(np.subtract(8*np.multiply(np.power(u_3, 4), u_4),
                                                        2*np.multiply(u_4, u_6)),
                                            np.power(u_2, 3)), 0) + \
                np.where(u_2 > 0, np.divide(np.power(u_4, 3), np.power(u_2, 4)), 0)
            u[u == 0] = 1e-6
            u = np.asarray(u).astype(np.float)
            u = np.divide(u, n)
            e = np.where(u >= 1e-5, np.sqrt(u), 0)
    return e


def centrality(arr1, arr2, cent_bins=np.array((20, 30, 40, 50, 60, 70, 80, 90, 95)),
               cent_bins_reverse=np.array((5, 10, 20, 30, 40, 50, 60, 70, 80)),
               reverse=False):
    if reverse is True:
        cent_bins = cent_bins_reverse
        arr2 = arr2 * 400
    n = np.sum(arr1, axis=1)
    arr = np.repeat(arr2[0][:-1].astype('int'), n.astype('int'))
    arr = np.hstack(arr)
    centralities = np.percentile(arr, cent_bins)
    if reverse is True:
        centralities = centralities * 0.0025
    return centralities


def cbwc(arr1, arr2, arr3, central, reverse=False):
    """
    This is for correcting for centrality bin width effects.
        Args:
            arr1: The MxN matrix of the correlation amounts from the 2D histogram.
            arr2: The XxY axis from the 2D hostogram.
            arr3: The centrality metric to cbwc
            central: The array of centrality cuts.
            reverse: False for RM based, True for b based.
    """
    arr3 = np.asarray(arr3).astype(float)
    arr3[np.isnan(arr3)] = 0
    n = np.sum(arr1, axis=1).astype('int')
    prot_num = arr2[0][:-1]
    C = []
    if reverse is True:
        central = central[::-1]
        index = np.hstack((np.where(prot_num > central[0])))
    else:
        index = np.hstack((np.where(prot_num <= central[0])))
    n_r = np.sum(n[index])
    arr = np.multiply(arr3[index], n[index])
    if n_r == 0:
        C.append(0)
    else:
        C.append(np.sum(arr) / n_r)
    for i in range(1, len(central)):
        if reverse is True:
            index = np.hstack((np.where((prot_num > central[i]) & (prot_num <= central[i-1]))))
        else:
            index = np.hstack((np.where((prot_num <= central[i]) & (prot_num > central[i-1]))))
        n_r = np.sum(n[index])
        arr = np.multiply(arr3[index], n[index])
        if n_r == 0:
            C.append(0)
        else:
            C.append(np.sum(arr)/n_r)
    if reverse is True:
        index = np.hstack((np.where(prot_num <= central[-1])))
    else:
        index = np.hstack((np.where(prot_num > central[-1])))
    n_r = np.sum(n[index])
    arr = np.multiply(arr3[index], n[index])
    if n_r == 0:
        C.append(0)
    else:
        C.append(np.sum(arr) / n_r)
    C = np.asarray(C)
    return C


def no_cbwc(arr1, arr2, arr3, central, reverse=False):
    """
    This is for showing what happens when you don't
    correct for centrality bin width effects.
        Args:
            arr1: The MxN matrix of the correlation amounts from the 2D histogram.
            arr2: The XxY axis from the 2D hostogram.
            arr3: The centrality metric to cbwc
            central: The array of centrality cuts.
            reverse: False for RM based, True for b based.
    """
    arr3 = np.asarray(arr3).astype(float)
    arr3[np.isnan(arr3)] = 0
    n = np.sum(arr1, axis=1).astype('int')
    prot_num = arr2[0][:-1]
    C = []
    if reverse is True:
        central = central[::-1]
        index = np.hstack((np.where(prot_num > central[0])))
    else:
        index = np.hstack((np.where(prot_num <= central[0])))
    arr = arr3[index]
    C.append(np.mean(arr))
    for i in range(1, len(central)):
        if reverse is True:
            index = np.hstack((np.where((prot_num > central[i]) & (prot_num <= central[i-1]))))
        else:
            index = np.hstack((np.where((prot_num <= central[i]) & (prot_num > central[i-1]))))
        arr = arr3[index]
        C.append(np.mean(arr))
    if reverse is True:
        index = np.hstack((np.where(prot_num <= central[-1])))
    else:
        index = np.hstack((np.where(prot_num > central[-1])))
    arr = arr3[index]
    C.append(np.mean(arr))
    C = np.asarray(C)
    return C


def cum_plot_int(ylabels, target, pro_bins, C, C_labels, xlabels, gev='200'):
    for k in range(4):
        fig, ax = plt.subplots(3, len(ylabels), figsize=(16, 9), constrained_layout=True)
        plt.suptitle(C_labels[k] + r' for $\sqrt{s_{NN}}$= ' + gev + ' GeV (UrQMD)',
                     fontsize=30)
        for i in range(3):
            for j in range(len(ylabels)):
                if target == 'b':
                    if j > 1:
                        ax[i, j].scatter(pro_bins[i][j][0][:-1],
                                          np.flip(C[k][i][j]),
                                          marker='.', s=5)
                        ax[i, j].set_xticks(np.linspace(0, pro_bins[i][j][0][:-1][-1], 5))
                        ax[i, j].set_xticklabels(np.flip(np.round(np.linspace(0,
                                                                              pro_bins[i][j][0][:-1][-1],
                                                                              5),
                                                                  1)).astype('str'),
                                                 rotation=45)
                    else:
                        ax[i, j].scatter(pro_bins[i][j][0][:-1], C[k][i][j], marker='.', s=5)
                        ax[i, j].set_xticks(np.linspace(0, pro_bins[i][j][0][:-1][-1], 5))
                        ax[i, j].set_xticklabels(np.linspace(0,
                                                             pro_bins[i][j][0][:-1][-1],
                                                             5).astype('int'),
                                                 rotation=45)
                else:
                    if j == 2:
                        ax[i, j].scatter(pro_bins[i][j][0][:-1],
                                          np.flip(C[k][i][j]),
                                          marker='.', s=5)
                        ax[i, j].set_xticks(np.linspace(0, pro_bins[i][j][0][:-1][-1], 5))
                        ax[i, j].set_xticklabels(np.flip(np.round(np.linspace(0,
                                                                              pro_bins[i][j][0][:-1][-1],
                                                                              5),
                                                                  1)).astype('str'),
                                                 rotation=45)
                    else:
                        ax[i, j].scatter(pro_bins[i][j][0][:-1], C[k][i][j], marker='.', s=5)
                        ax[i, j].set_xticks(np.linspace(0, pro_bins[i][j][0][:-1][-1], 5))
                        ax[i, j].set_xticklabels(np.linspace(0,
                                                             pro_bins[i][j][0][:-1][-1],
                                                             5).astype('int'),
                                                 rotation=45)
                ax[i, j].set_xlabel(ylabels[j])
                ax[i, j].set_ylabel(r'<' + xlabels[i] + r'>')
        plt.show()
        plt.close()


def cum_plot_int_err(ylabels, target, pro_bins, C, E, C_labels, xlabels, gev='200'):
    for k in range(4):
        fig, ax = plt.subplots(3, len(ylabels), figsize=(16, 9), constrained_layout=True)
        plt.suptitle(C_labels[k] + r' for $\sqrt{s_{NN}}$= ' + gev + ' GeV (UrQMD)',
                     fontsize=30)
        for i in range(3):
            for j in range(len(ylabels)):
                if target == 'b':
                    if j > 1:
                        ax[i, j].errorbar(pro_bins[i][j][0][:-1],
                                          np.flip(C[k][i][j]),
                                          yerr=np.flip(E[k][i][j]),
                                          marker='.', ms=3, lw=0,
                                          elinewidth=1, capsize=2)
                        ax[i, j].set_xticks(np.linspace(0, pro_bins[i][j][0][:-1][-1], 5))
                        ax[i, j].set_xticklabels(np.flip(np.round(np.linspace(0,
                                                                              pro_bins[i][j][0][:-1][-1],
                                                                              5),
                                                                  1)).astype('str'),
                                                 rotation=45)
                    else:
                        ax[i, j].errorbar(pro_bins[i][j][0][:-1], C[k][i][j], yerr=E[k][i][j],
                                          marker='.', ms=3, lw=0, elinewidth=1, capsize=2)
                        ax[i, j].set_xticks(np.linspace(0, pro_bins[i][j][0][:-1][-1], 5))
                        ax[i, j].set_xticklabels(np.linspace(0,
                                                             pro_bins[i][j][0][:-1][-1],
                                                             5).astype('int'),
                                                 rotation=45)
                else:
                    if j == 2:
                        ax[i, j].errorbar(pro_bins[i][j][0][:-1],
                                          np.flip(C[k][i][j]),
                                          yerr=np.flip(E[k][i][j]),
                                          marker='.', ms=3, lw=0,
                                          elinewidth=1, capsize=2)
                        ax[i, j].set_xticks(np.linspace(0, pro_bins[i][j][0][:-1][-1], 5))
                        ax[i, j].set_xticklabels(np.flip(np.round(np.linspace(0,
                                                                              pro_bins[i][j][0][:-1][-1],
                                                                              5),
                                                                  1)).astype('str'),
                                                 rotation=45)
                    else:
                        ax[i, j].errorbar(pro_bins[i][j][0][:-1], C[k][i][j], yerr=E[k][i][j],
                                          marker='.', ms=3, lw=0, elinewidth=1, capsize=2)
                        ax[i, j].set_xticks(np.linspace(0, pro_bins[i][j][0][:-1][-1], 5))
                        ax[i, j].set_xticklabels(np.linspace(0,
                                                             pro_bins[i][j][0][:-1][-1],
                                                             5).astype('int'),
                                                 rotation=45)
                ax[i, j].set_xlabel(ylabels[j])
                ax[i, j].set_ylabel(r'<' + xlabels[i] + r'>')
        plt.show()
        plt.close()


def cum_plot_int_err_test(ylabels, target, pro_bins, C, Cu, Cd, C_labels, xlabels, gev='200'):
    for k in range(4):
        fig, ax = plt.subplots(3, len(ylabels), figsize=(16, 9), constrained_layout=True)
        plt.suptitle(C_labels[k] + r' for $\sqrt{s_{NN}}$= ' + gev + ' GeV (UrQMD)',
                     fontsize=30)
        for i in range(3):
            for j in range(len(ylabels)):
                if target == 'b':
                    if j > 1:
                        ax[i, j].errorbar(pro_bins[i][j][0][:-1],
                                          np.flip(C[k][i][j]),
                                          yerr=np.flip((Cd[k][i][j], Cu[k][i][j])),
                                          marker='.', ms=3, lw=0,
                                          elinewidth=1, capsize=2)
                        ax[i, j].set_xticks(np.linspace(0, pro_bins[i][j][0][:-1][-1], 5))
                        ax[i, j].set_xticklabels(np.flip(np.round(np.linspace(0,
                                                                              pro_bins[i][j][0][:-1][-1],
                                                                              5),
                                                                  1)).astype('str'),
                                                 rotation=45)
                    else:
                        ax[i, j].errorbar(pro_bins[i][j][0][:-1], C[k][i][j], yerr=(Cd[k][i][j], Cu[k][i][j]),
                                          marker='.', ms=3, lw=0, elinewidth=1, capsize=2)
                        ax[i, j].set_xticks(np.linspace(0, pro_bins[i][j][0][:-1][-1], 5))
                        ax[i, j].set_xticklabels(np.linspace(0,
                                                             pro_bins[i][j][0][:-1][-1],
                                                             5).astype('int'),
                                                 rotation=45)
                else:
                    if j == 2:
                        ax[i, j].errorbar(pro_bins[i][j][0][:-1],
                                          np.flip(C[k][i][j]),
                                          yerr=np.flip((Cd[k][i][j], Cu[k][i][j])),
                                          marker='.', ms=3, lw=0,
                                          elinewidth=1, capsize=2)
                        ax[i, j].set_xticks(np.linspace(0, pro_bins[i][j][0][:-1][-1], 5))
                        ax[i, j].set_xticklabels(np.flip(np.round(np.linspace(0,
                                                                              pro_bins[i][j][0][:-1][-1],
                                                                              5),
                                                                  1)).astype('str'),
                                                 rotation=45)
                    else:
                        ax[i, j].errorbar(pro_bins[i][j][0][:-1], C[k][i][j], yerr=(Cd[k][i][j], Cu[k][i][j]),
                                          marker='.', ms=3, lw=0, elinewidth=1, capsize=2)
                        ax[i, j].set_xticks(np.linspace(0, pro_bins[i][j][0][:-1][-1], 5))
                        ax[i, j].set_xticklabels(np.linspace(0,
                                                             pro_bins[i][j][0][:-1][-1],
                                                             5).astype('int'),
                                                 rotation=45)
                ax[i, j].set_xlabel(ylabels[j])
                ax[i, j].set_ylabel(r'<' + xlabels[i] + r'>')
        plt.show()
        plt.close()


def cn_data(arr, power=1):
    if power <= 3:
        u = arr[power]
    else:
        u = arr[power] + 3*(arr[1]**2)
    return u


def en_data(arr, n, power=0):
    if power == 0:
        e = np.sqrt(arr[1]/n)
    elif power == 1:
        e = np.sqrt((arr[3] - arr[1]**2)/n)
    elif power == 2:
        e = np.sqrt((9*(arr[1]**3) - 6*arr[1]*arr[3] - arr[2]**3 + arr[5])/n)
    else:
        e = np.sqrt((-36*(arr[1]**4) + 48*arr[3]*(arr[1]**2) + 64*arr[1]*(arr[2]**2) -
                     12*arr[1]*arr[5] - 8*arr[2]*arr[4] - arr[3]**2 + arr[7])/n)
    return e


def ern_data(arr, n, power=1):
    if power == 0:
        e = np.sqrt(arr[1] / n)
    elif power == 1:
        e = ((arr[3] - (arr[1]**2))/(arr[0]**2) - (2*arr[1]*arr[2])/(arr[0]**3) +
             (arr[1]**3)/(arr[0]**4))/n
    elif power == 2:
        e = (9*arr[1] - 6*arr[3]/arr[1] + 6*(arr[2]**2)/(arr[1]**2) + arr[5]/(arr[1]**2) -
             (2*arr[2]*arr[4])/(arr[1]**3) + ((arr[2]**2)*arr[3])/(arr[1]**4))/n
    else:
        e = (-9*(arr[1]**2) + 9*arr[3] + (40*(arr[2]**2) - 6*arr[5])/arr[1] -
             (8*arr[2]*arr[4] + 6*(arr[3]**2) + arr[7])/(arr[1]**2) +
             (8*(arr[2]**2)*arr[3] - 2*arr[3]*arr[5])/(arr[1]**3) +
             (arr[3]**3)/(arr[1]**3))/n
    return e


def cbwc_data(arr, n_cent, x, central):
    C = []
    arr = np.asarray(arr)
    arr[np.isnan(arr)] = 100
    n_cent = np.asarray(n_cent)
    index = (x <= central[0])
    arr_ = arr[index]
    n_cent_ = n_cent[index]
    n = np.sum(n_cent_)
    C.append(np.sum(np.multiply(n_cent_, arr_))/n)
    for i in range(1, len(central)):
        index = ((x <= central[i]) & (x > central[i-1]))
        arr_ = arr[index]
        n_cent_ = n_cent[index]
        n = np.sum(n_cent_)
        C.append(np.sum(np.multiply(n_cent_, arr_)) / n)
    index = (x > central[-1])
    arr_ = arr[index]
    n_cent_ = n_cent[index]
    n = np.sum(n_cent_)
    C.append(np.sum(np.multiply(n_cent_, arr_))/n)
    return C


def cum_plot_int_data(ylabels, x, C, C_labels, xlabels, gev='14.6'):
    markers = ['o', '8', 's', 'P', '*', 'X', 'D']
    color = ['orangered', 'orange', 'black', 'blue', 'purple', 'darkviolet', 'deepskyblue']
    for k in range(4):
        fig, ax = plt.subplots(3, len(ylabels), figsize=(16, 9), constrained_layout=True)
        plt.suptitle(C_labels[k] + r' for $\sqrt{s_{NN}}$= ' + gev + ' GeV (STAR data)',
                     fontsize=30)
        for i in range(3):
            for j in range(len(ylabels)):
                ax[i, j].scatter(x[j], C[j][i][k], marker=markers[j], s=5, color=color[j])
                # ax[i, j].set_xticks(np.linspace(0, x[j], 5))
                # ax[i, j].set_xticklabels(np.linspace(0, x[j], 5).astype('int'), rotation=45)
                ax[i, j].set_xlabel(ylabels[j])
                ax[i, j].set_ylabel(r'<' + xlabels[i] + r'>')
        plt.show()
        plt.close()


def cum_plot_int_err_data(ylabels, x, C, C_labels, E, xlabels, gev='14.6'):
    markers = ['o', '8', 's', 'P', '*', 'X', 'D']
    color = ['orangered', 'orange', 'black', 'blue', 'purple', 'darkviolet', 'deepskyblue']
    for k in range(4):
        fig, ax = plt.subplots(3, len(ylabels), figsize=(16, 9), constrained_layout=True)
        plt.suptitle(C_labels[k] + r' for $\sqrt{s_{NN}}$= ' + gev + ' GeV (STAR data)',
                     fontsize=30)
        for i in range(3):
            for j in range(len(ylabels)):
                C[j][i][k] = np.asarray(C[j][i][k])
                E[j][i][k] = np.asarray(E[j][i][k])
                ax[i, j].fill_between(x[j], np.add(C[j][i][k], E[j][i][k]),
                                      np.subtract(C[j][i][k], E[j][i][k]), facecolor=color[j],
                                      alpha=0.2)
                ax[i, j].scatter(x[j], C[j][i][k], marker=markers[j], s=5, color=color[j])
                # ax[i, j].set_xticks(np.linspace(0, x[j], 5))
                # ax[i, j].set_xticklabels(np.linspace(0, x[j], 5).astype('int'), rotation=45)
                ax[i, j].set_xlabel(ylabels[j])
                ax[i, j].set_ylabel(r'<' + xlabels[i] + r'>')
        plt.show()
        plt.close()
