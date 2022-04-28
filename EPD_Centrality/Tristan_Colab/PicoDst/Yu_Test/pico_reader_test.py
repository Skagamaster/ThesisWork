#
# \PicoDst reader for Python
#
# \author Skipper Kagamaster
# \date 03/19/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
# /

# Not all these are used right now.
import numpy as np
import pandas as pd
import uproot as up
import awkward as ak
from scipy.signal import savgol_filter as sgf
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import os

# Speed of light, in m/s
SPEED_OF_LIGHT = 299792458
# Proton mass, in GeV
PROTON_MASS = 0.9382720813

run_list = np.loadtxt(r'C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\runs.txt', delimiter=',').astype("int")
sig = np.loadtxt(r'C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\nsigmaVals.txt', delimiter=',')  # [[447, 669]]


# This appears to no longer be needed.
def ak_to_numpy_flat(*args):
    """This transforms an awkward array into a flat, numpy array for plotting.
        Args:
            *args: The array(s) to flatten and go numpy."""
    for arg in args:
        arg = ak.to_numpy(ak.flatten(arg))
        yield arg


# This appears to no longer be needed.
def ak_to_numpy(*args):
    """This transforms multiple awkward arrays into numpy arrays for plotting.
        Args:
            *args: The array(s) to go numpy."""
    for arg in args:
        arg = ak.to_numpy(arg)
        yield arg


def index_cut(a, *args):
    for arg in args:
        arg = arg[a]
        yield arg


# This appears to no longer be needed.
def get_ave(arr):
    """This gives the mean and standard error for a numpy array.
        Args:
            arr: Array you want to get the mean and std error from"""
    ave = np.mean(arr)
    dev = np.divide(np.std(arr), len(arr))
    return ave, dev


# TODO Check to see that these results are reasonable. Graph against p_t.
def rapidity(p_z, p_g):
    e_p = np.power(np.add(PROTON_MASS ** 2, np.power(p_g, 2)), 1 / 2)
    y = 0.5 * np.log(np.divide(e_p + p_z, e_p - p_z))
    return y


class EPD_Hits:
    mID = None
    mQT_data = None
    mnMip = None
    position = None  # Supersector position on wheel [1, 12]
    tiles = None  # Tile number on the Supersector [1, 31]
    row = None  # Row Number [1, 16]
    EW = None  # -1 for East wheel, +1 for West wheel
    ADC = None  # ADC Value reported by QT board [0, 4095]
    TAC = None  # TAC value reported by QT board[0, 4095]
    TDC = None  # TDC value reported by QT board[0, 32]
    has_TAC = None  # channel has a TAC
    nMip = None  # gain calibrated signal, energy loss in terms of MPV of Landau convolution for a MIP
    status_is_good = None  # good status, according to database

    # def __init__(self, mID, mQT_data, mnMips, lower_bound=0.2, upper_bound=3, day=94):
    def __init__(self, mID, mQT_data, lower_bound=0.2, upper_bound=3, day=94):

        self.mID = mID
        self.mQT_data = mQT_data
        # self.mnMip = mnMips
        # For entering calibrations
        self.day = day
        self.days = np.array([94, 105, 110, 113, 114, 123, 138, 139])
        self.data_day = str(self.days[np.where(self.days <= self.day)[0][-1]])
        calfile = np.loadtxt(r'D:\14GeV\ChiFit\Nmip_Day_' + self.data_day + '.txt')[:, 4]

        self.has_TAC = np.bitwise_and(np.right_shift(self.mQT_data, 29), 0x1)
        self.status_is_good = np.bitwise_and(np.right_shift(self.mQT_data, 30), 0x1)

        self.adc = np.bitwise_and(self.mQT_data, 0x0FFF)
        # self.tac = np.bitwise_and(np.right_shift(self.mQT_data, 12), 0x0FFF)
        # self.TDC = np.bitwise_and(np.right_shift(self.mQT_data, 24), 0x001F)

        # Trying to speed things up a bit, so let's flatten and then unflatten to
        # broadcast instead of looping.
        counts = ak.num(self.adc)
        self.EW = ak.Array(np.sign(self.mID))
        self.EW = ak.where(self.EW > 0, 1, 0)
        self.position = np.abs(self.mID // 100)
        self.tiles = np.abs(self.mID) % 100
        self.row = ((np.abs(self.mID) % 100) // 2) + 1
        calNmip = ak.flatten(self.EW * 372 + (self.position - 1) * 31 + self.tiles - 1)
        ADC_cals = ak.Array(calfile[calNmip])
        ADC_cals = ak.unflatten(ADC_cals, counts)
        self.nMip = ak.where(self.status_is_good, self.adc, 0)/ADC_cals
        # nMIP truncation
        self.nMip = ak.where(self.nMip <= lower_bound, lower_bound, self.nMip)
        self.nMip = ak.where(self.nMip >= upper_bound, upper_bound, self.nMip)

    def generate_epd_hit_matrix(self):
        ring_sum = np.zeros((32, len(self.nMip)))
        # print("Filling array of dimension", ring_sum.shape)
        for i in range(32):
            x = i % 16
            if i < 16:
                ew_mask = ((self.EW > 0) & (self.row == x + 1))
            else:
                ew_mask = ((self.EW < 0) & (self.row == x + 1))
            ring_i = ak.sum(self.nMip[ew_mask], axis=-1)
            ring_sum[i] = ring_i
        return ring_sum.T


class PicoDST:
    """This class makes the PicoDST from the root file, along with
    all of the observables I use for proton kurtosis analysis."""

    def __init__(self, data_file=None):
        """This defines the variables we'll be using
        in the class."""
        self.data: bool
        self.num_events = None
        self.v_x = None
        self.v_y = None
        self.v_z = None
        self.v_r = None
        self.vz_vpd = None
        self.refmult3 = None
        self.tofmult = None
        self.tofmatch = None
        self.beta_eta_1 = None
        self.epd_hits = None
        self.zdcx = None

        self.p_t = None
        self.p_g = None
        self.phi = None
        self.dca = None
        self.eta = None
        self.nhitsfit = None
        self.nhitsdedx = None
        self.dedx = None
        self.rapidity = None
        self.nhitsmax = None
        self.nsigma_proton = None
        self.tofpid = None
        self.m_2 = None
        self.charge = None
        self.beta = None

        self.p_t_tof = None
        self.p_g_tof = None
        self.phi_tof = None
        self.dca_tof = None
        self.eta_tof = None
        self.nhitsfit_tof = None
        self.nhitsdedx_tof = None
        self.dedx_tof = None
        self.rapidity_tof = None
        self.nhitsmax_tof = None
        self.nsigma_proton_tof = None
        self.charge_tof = None

        self.protons_low = None
        self.antiprotons_low = None
        self.protons_high = None
        self.antiprotons_high = None
        self.dedx_histo = None
        self.p_g_histo = None
        self.charge_histo = None
        self.run_id = None
        self.event_df = None
        self.track_df = None
        self.toftrack_df = None
        self.day = None
        self.ave_df = None

        if data_file is not None:
            self.import_data(data_file)

    def import_data(self, data_in):
        """This imports the data. You must have the latest versions
        of uproot and awkward installed on your machine (uproot4 and
        awkward 1.0 as of the time of this writing).
        Use pip install uproot awkward.
        Args:
            data_in (str): The path to the picoDst ROOT file"""
        try:
            with up.open(data_in) as data:
                self.num_events = len(data["PicoDst"]["Event"]["Event.mPrimaryVertexX"].array())
                self.run_id = int(ak.to_numpy(ak.flatten(data["PicoDst"]["Event"]["Event.mRunId"].array()))[0])
                self.day = int(str(self.run_id)[2:5])
                # Make vertices

                self.v_x = ak.to_numpy(ak.flatten(data["PicoDst"]["Event"]["Event.mPrimaryVertexX"].array()))
                self.v_y = ak.to_numpy(ak.flatten(data["PicoDst"]["Event"]["Event.mPrimaryVertexY"].array()))
                self.v_z = ak.to_numpy(ak.flatten(data["PicoDst"]["Event"]["Event.mPrimaryVertexZ"].array()))
                self.v_r = np.sqrt(np.power(self.v_x, 2) + np.power(self.v_y, 2))
                self.vz_vpd = ak.to_numpy(ak.flatten(data["PicoDst"]["Event"]["Event.mVzVpd"].array()))
                self.zdcx = ak.to_numpy(ak.flatten(data["PicoDst"]["Event"]["Event.mZDCx"].array()))

                self.refmult3 = ak.to_numpy(
                    ak.flatten(data["PicoDst"]["Event"]["Event.mRefMult3PosEast"].array() +
                               data["PicoDst"]["Event"]["Event.mRefMult3PosWest"].array() +
                               data["PicoDst"]["Event"]["Event.mRefMult3NegEast"].array() +
                               data["PicoDst"]["Event"]["Event.mRefMult3NegWest"].array()))
                self.tofmult = ak.to_numpy(
                    ak.flatten(data["PicoDst"]["Event"]["Event.mbTofTrayMultiplicity"].array()))
                # Make p_g and p_t
                p_x = data["PicoDst"]["Track"]["Track.mGMomentumX"].array()
                p_y = data["PicoDst"]["Track"]["Track.mGMomentumY"].array()
                p_y = ak.where(p_y == 0.0, 1e-10, p_y)  # to avoid infinities
                p_z = data["PicoDst"]["Track"]["Track.mGMomentumZ"].array()
                self.p_t = np.sqrt(np.power(p_x, 2) + np.power(p_y, 2))
                self.p_g = np.sqrt((np.power(p_x, 2) + np.power(p_y, 2) + np.power(p_z, 2)))
                self.eta = np.arcsinh(np.divide(p_z, self.p_t))
                # Make dca
                dca_x = data["PicoDst"]["Track"]["Track.mOriginX"].array() - self.v_x
                dca_y = data["PicoDst"]["Track"]["Track.mOriginY"].array() - self.v_y
                dca_z = data["PicoDst"]["Track"]["Track.mOriginZ"].array() - self.v_z
                self.dca = np.sqrt((np.power(dca_x, 2) + np.power(dca_y, 2) + np.power(dca_z, 2)))
                self.nhitsfit = data["PicoDst"]["Track"]["Track.mNHitsFit"].array()
                self.beta = data["PicoDst"]["BTofPidTraits"]["BTofPidTraits.mBTofBeta"].array() / 20000.0
                self.beta = ak.where(self.beta == 0.0, 1e-10, self.beta)  # To avoid infinities
                self.tofpid = data["PicoDst"]["BTofPidTraits"]["BTofPidTraits.mTrackIndex"].array()
                # Make B_n_1
                be1_1 = ak.where(self.beta > 0.1, 1, 0)
                be1_2 = ak.where(np.absolute(self.eta[self.tofpid]) < 1.0, 1, 0)
                be1_3 = ak.where(self.dca[self.tofpid] < 3.0, 1, 0)
                be1_4 = ak.where(np.absolute(self.nhitsfit[self.tofpid]) > 10, 1, 0)
                be1 = be1_1 * be1_2 * be1_3 * be1_4
                self.beta_eta_1 = ak.to_numpy(ak.sum(be1, axis=-1))
                # Make bTOFmatch
                nTofMatch = data["PicoDst"]["Event"]["Event.mNBTOFMatch"].array()
                nTofMatch1 = ak.flatten(ak.where(nTofMatch > 0, 1, 0))
                nTofMatch2 = ak.where(abs(self.eta) < 0.5, 1, 0)
                nTofMatch3 = ak.where(abs(self.dca) < 3.0, 1, 0)
                nTofMatch4 = ak.where(self.nhitsfit > 10, 1, 0)
                counts = ak.num(nTofMatch2)
                nTofMatch2 = ak.flatten(nTofMatch2)
                nTofMatch3 = ak.flatten(nTofMatch3)
                nTofMatch4 = ak.flatten(nTofMatch4)
                nTofMatchTot = nTofMatch2 * nTofMatch3 * nTofMatch4
                nTofMatchTot = ak.unflatten(nTofMatchTot, counts) * nTofMatch1
                self.tofmatch = ak.to_numpy(ak.sum(nTofMatchTot, axis=-1))

                self.rapidity = rapidity(p_z, self.p_g)
                self.nhitsdedx = data["PicoDst"]["Track"]["Track.mNHitsDedx"].array()
                self.nhitsmax = data["PicoDst"]["Track"]["Track.mNHitsMax"].array()
                self.nhitsmax = ak.where(self.nhitsmax == 0, 1e-10, self.nhitsmax)  # to avoid infinities
                self.dedx = data["PicoDst"]["Track"]["Track.mDedx"].array()
                self.nsigma_proton = data["PicoDst"]["Track"]["Track.mNSigmaProton"].array()
                self.charge = ak.where(self.nhitsfit >= 0, 1, -1)
                # Make m^2
                p_squared = np.power(self.p_g[self.tofpid], 2)
                b_squared = ak.where(self.beta == 0.0, 1e-10, self.beta)  # to avoid infinities
                b_squared = np.divide(1, np.power(b_squared, 2)) - 1
                self.m_2 = np.multiply(p_squared, b_squared)
                # Make phi.
                o_x = data["PicoDst"]["Track"]["Track.mOriginX"].array()
                o_y = data["PicoDst"]["Track"]["Track.mOriginY"].array()
                self.phi = np.arctan2(o_y, o_x)

                self.p_t_tof, self.p_g_tof, self.phi_tof, self.dca_tof, self.eta_tof, \
                    self.nhitsfit_tof, self.nhitsdedx_tof, self.dedx_tof, self.rapidity_tof, self.nhitsmax_tof, \
                    self.nsigma_proton_tof, self.charge_tof = \
                    index_cut(self.tofpid, self.p_t, self.p_g, self.phi, self.dca,
                              self.eta, self.nhitsfit, self.nhitsdedx, self.dedx, self.rapidity,
                              self.nhitsmax, self.nsigma_proton, self.charge)

                # Load EPD Data
                epd_hit_id_data = data["PicoDst"]["EpdHit"]["EpdHit.mId"].array()
                epd_hit_mQTdata = data["PicoDst"]["EpdHit"]["EpdHit.mQTdata"].array()
                # epd_hit_mnMIP = data["PicoDst"]["EpdHit"]["EpdHit.mnMIP"].array()
                self.epd_hits = EPD_Hits(epd_hit_id_data, epd_hit_mQTdata, day=self.day)
                self.epd_hits = self.epd_hits.generate_epd_hit_matrix()

        # except ValueError:  # Skip empty picos.
        #     print("ValueError at: " + data_in)  # Identifies the misbehaving file.
        except KeyError:  # Skip non empty picos that have no data.
            print("KeyError at: " + data_in)  # Identifies the misbehaving file.

    def event_cuts(self, v_r_cut=2.0, v_z_cut=30.0, v_cut=1.0e-5,
                   tofmult_refmult=np.array([[2.536, 200], [1.352, -54.08]]),
                   tofmatch_refmult=np.array([0.239, -14.34]), beta_refmult=np.array([0.447, -17.88])):
        index = ((np.absolute(self.v_z) <= v_z_cut) & (self.v_r < v_r_cut) &
                 (self.tofmult <= (np.multiply(tofmult_refmult[0][0], self.refmult3) + tofmult_refmult[0][1])) &
                 (self.tofmult >= (np.multiply(tofmult_refmult[1][0], self.refmult3) + tofmult_refmult[1][1])) &
                 (self.tofmatch >= (np.multiply(tofmatch_refmult[0], self.refmult3) + tofmatch_refmult[1])) &
                 (self.beta_eta_1 >= (np.multiply(beta_refmult[0], self.refmult3) + beta_refmult[1])) &
                 (abs(self.v_x) >= v_cut) & (abs(self.v_x) >= v_cut) & (abs(self.v_y) >= v_cut))
        self.v_x, self.v_y, self.v_z, self.v_r, self.refmult3, self.tofmult, \
            self.tofmatch, self.beta_eta_1, self.p_t, self.p_g, self.phi, self.dca, \
            self.eta, self.nhitsfit, self.nhitsdedx, self.m_2, self.charge, self.beta, \
            self.dedx, self.zdcx, self.rapidity, self.nhitsmax, self.nsigma_proton, \
            self.tofpid, self.epd_hits, self.vz_vpd, self.p_t_tof, self.p_g_tof, self.phi_tof, \
            self.dca_tof, self.eta_tof, self.nhitsfit_tof, self.nhitsdedx_tof, self.dedx_tof, \
            self.rapidity_tof, self.nhitsmax_tof, self.nsigma_proton_tof, self.charge_tof = \
            index_cut(index, self.v_x, self.v_y, self.v_z, self.v_r, self.refmult3, self.tofmult,
                      self.tofmatch, self.beta_eta_1, self.p_t, self.p_g, self.phi, self.dca,
                      self.eta, self.nhitsfit, self.nhitsdedx, self.m_2, self.charge, self.beta,
                      self.dedx, self.zdcx, self.rapidity, self.nhitsmax, self.nsigma_proton,
                      self.tofpid, self.epd_hits, self.vz_vpd, self.p_t_tof, self.p_g_tof, self.phi_tof,
                      self.dca_tof, self.eta_tof, self.nhitsfit_tof, self.nhitsdedx_tof, self.dedx_tof,
                      self.rapidity_tof, self.nhitsmax_tof, self.nsigma_proton_tof, self.charge_tof)

    def track_qa_cuts(self, nhitsdedx_cut=5, nhitsfit_cut=20, ratio=0.52, dca_cut=1.0,
                      pt_low_cut=0.4, pt_high_cut=2.0, rapid_cut=0.5):
        index = ((self.nhitsdedx > nhitsdedx_cut) & (np.absolute(self.nhitsfit) > nhitsfit_cut) &
                 (np.divide(1 + np.absolute(self.nhitsfit), 1 + self.nhitsmax) > ratio) &
                 (self.dca < dca_cut) & (self.p_t > pt_low_cut) &
                 (self.p_t < pt_high_cut) & (np.absolute(self.rapidity) <= rapid_cut))
        self.p_t, self.p_g, self.phi, self.dca, self.eta, self.nhitsfit, self.nhitsdedx, self.charge, self.dedx, \
            self.rapidity, self.nhitsmax, self.nsigma_proton = \
            index_cut(index, self.p_t, self.p_g, self.phi, self.dca, self.eta, self.nhitsfit, self.nhitsdedx,
                      self.charge, self.dedx, self.rapidity, self.nhitsmax, self.nsigma_proton)
        index_tof = ((self.nhitsdedx_tof > nhitsdedx_cut) & (np.absolute(self.nhitsfit_tof) > nhitsfit_cut) &
                     (np.divide(1 + np.absolute(self.nhitsfit_tof), 1 + self.nhitsmax_tof) > ratio) &
                     (self.dca_tof < dca_cut) & (self.p_t_tof > pt_low_cut) &
                     (self.p_t_tof < pt_high_cut) & (np.absolute(self.rapidity_tof) <= rapid_cut))
        self.p_t_tof, self.p_g_tof, self.phi_tof, self.dca_tof, self.eta_tof, self.nhitsfit_tof, \
            self.nhitsdedx_tof, self.charge_tof, self.dedx_tof, self.rapidity_tof, self.nhitsmax_tof, \
            self.nsigma_proton_tof, self.beta, self.m_2 = \
            index_cut(index_tof, self.p_t_tof, self.p_g_tof, self.phi_tof, self.dca_tof, self.eta_tof,
                      self.nhitsfit_tof, self.nhitsdedx_tof, self.charge_tof, self.dedx_tof,
                      self.rapidity_tof, self.nhitsmax_tof, self.nsigma_proton_tof, self.beta, self.m_2)

    # This appears to no longer be needed.
    def track_qa_cuts_tof(self, nhitsdedx_cut=5, nhitsfit_cut=20, ratio=0.52, dca_cut=1.0,
                          pt_low_cut=0.4, pt_high_cut=2.0, rapid_cut=0.5):
        self.p_t, self.p_g, self.phi, self.dca, self.eta, self.nhitsfit, self.nhitsdedx, self.charge, self.dedx, \
            self.rapidity, self.nhitsmax, self.nsigma_proton = \
            index_cut(self.tofpid, self.p_t, self.p_g, self.phi, self.dca, self.eta, self.nhitsfit, self.nhitsdedx,
                      self.charge, self.dedx, self.rapidity, self.nhitsmax, self.nsigma_proton)
        index = ((self.nhitsdedx > nhitsdedx_cut) & (np.absolute(self.nhitsfit) > nhitsfit_cut) &
                 (np.divide(1 + np.absolute(self.nhitsfit), 1 + self.nhitsmax) > ratio) &
                 (self.dca < dca_cut) & (self.p_t > pt_low_cut) &
                 (self.p_t < pt_high_cut) & (np.absolute(self.rapidity) <= rapid_cut))
        self.p_t, self.p_g, self.phi, self.dca, self.eta, self.nhitsfit, self.nhitsdedx, self.charge, self.dedx, \
            self.rapidity, self.nhitsmax, self.nsigma_proton, self.beta, self.m_2 = \
            index_cut(index, self.p_t, self.p_g, self.phi, self.dca, self.eta, self.nhitsfit, self.nhitsdedx,
                      self.charge, self.dedx, self.rapidity, self.nhitsmax, self.nsigma_proton, self.beta, self.m_2)

    def select_protons(self, nsigma_cut=2000.0, pt_low_cut=0.8, pg_cut_low=1.0,
                           pt_high_cut=0.8, pg_high_cut=3.0, mass_low_cut=0.6, mass_high_cut=1.2):
        index = ((np.abs(self.nsigma_proton) <= nsigma_cut) & (self.p_t < pt_low_cut) & (self.p_g <= pg_cut_low))
        self.p_t, self.p_g, self.phi, self.dca, self.eta, self.nhitsfit, self.nhitsdedx, self.charge, self.dedx, \
            self.rapidity, self.nhitsmax, self.nsigma_proton = \
            index_cut(index, self.p_t, self.p_g, self.phi, self.dca, self.eta, self.nhitsfit, self.nhitsdedx,
                      self.charge, self.dedx, self.rapidity, self.nhitsmax, self.nsigma_proton)
        protons = ak.where(self.charge > 0, 1, 0)
        antiprotons = ak.where(self.charge < 0, 1, 0)
        self.protons_low = ak.sum(protons, axis=-1)
        self.antiprotons_low = ak.sum(antiprotons, axis=-1)

        index = ((np.abs(self.nsigma_proton_tof) <= nsigma_cut) & (self.p_t_tof >= pt_high_cut) &
                 (self.p_g_tof <= pg_high_cut) & (self.m_2 >= mass_low_cut) & (self.m_2 <= mass_high_cut))
        self.p_t_tof, self.p_g_tof, self.phi_tof, self.dca_tof, self.eta_tof, self.nhitsfit_tof, \
            self.nhitsdedx_tof, self.charge_tof, self.dedx_tof, self.rapidity_tof, self.nhitsmax_tof, \
            self.nsigma_proton, self.m_2, self.beta = \
            index_cut(index, self.p_t_tof, self.p_g_tof, self.phi_tof, self.dca_tof, self.eta_tof,
                      self.nhitsfit_tof, self.nhitsdedx_tof, self.charge_tof, self.dedx_tof,
                      self.rapidity_tof, self.nhitsmax_tof, self.nsigma_proton_tof, self.m_2, self.beta)
        protons = ak.where(self.charge_tof > 0, 1, 0)
        antiprotons = ak.where(self.charge_tof < 0, 1, 0)
        self.protons_high = ak.sum(protons, axis=-1)
        self.antiprotons_high = ak.sum(antiprotons, axis=-1)

    # This appears to no longer be necessary.
    def select_protons_high(self, nsigma_cut=2000.0, pt_low_cut=0.8, pg_cut=3.0,
                            mass_low_cut=0.6, mass_high_cut=1.2):
        index = ((np.abs(self.nsigma_proton_tof) <= nsigma_cut) & (self.p_t_tof >= pt_low_cut) &
                 (self.p_g_tof <= pg_cut) & (self.m_2 >= mass_low_cut) & (self.m_2 <= mass_high_cut))
        self.p_t_tof, self.p_g_tof, self.phi_tof, self.dca_tof, self.eta_tof, self.nhitsfit_tof, \
            self.nhitsdedx_tof, self.charge_tof, self.dedx_tof, self.rapidity_tof, self.nhitsmax_tof, \
            self.nsigma_proton = \
            index_cut(index, self.p_t_tof, self.p_g_tof, self.phi_tof, self.dca_tof, self.eta_tof,
                      self.nhitsfit_tof, self.nhitsdedx_tof, self.charge_tof, self.dedx_tof,
                      self.rapidity_tof, self.nhitsmax_tof, self.nsigma_proton_tof)
        protons = ak.where(self.charge_tof > 0, 1, 0)
        antiprotons = ak.where(self.charge_tof < 0, 1, 0)
        self.protons_high = ak.sum(protons, axis=-1)
        self.antiprotons_high = ak.sum(antiprotons, axis=-1)

    def calibrate_nsigmaproton(self):
        """Calibration of nSigmaProton for 0.0 < p_t < 0.8 (assumed 0 otherwise)"""
        # First, we'll separate it into discrete groupings of p_t.
        sig_length = 19
        nsigmaproton_p = []
        p_t_n = ak.to_numpy(ak.flatten(self.p_t))
        ns_n = ak.to_numpy(ak.flatten(self.nsigma_proton))
        nsigmaproton_p.append(ns_n[(p_t_n <= 0.2)])
        for k in range(2, sig_length + 1):
            nsigmaproton_p.append(ns_n[((p_t_n > 0.1 * k) & (p_t_n <= 0.1 * (k + 1)))])
        nsigmaproton_p = np.array(nsigmaproton_p)

        # Now to find the peak of the proton distribution. I'm going to try smoothing the
        # distributions, then finding the inflection points via a second order derivative.
        sig_means = []
        p_count = 0
        for dist in nsigmaproton_p:
            counter, bins = np.histogram(dist, range=(-10000, 10000), bins=200)
            sgf_proton_3 = sgf(counter, 45, 2)
            sgf_proton_3_2 = sgf(sgf_proton_3, 45, 2, deriv=2)
            infls = bins[:-1][np.where(np.diff(np.sign(sgf_proton_3_2)))[0]]
            sig_mean = 0
            if infls.size >= 2:
                infls_bounds = np.sort(np.absolute(infls))
                first = infls[np.where(np.absolute(infls) == infls_bounds[0])[0][0]]
                second = infls[np.where(np.absolute(infls) == infls_bounds[1])[0][0]]
                if first > second:
                    sig_mean = first - (first - second) / 2
                else:
                    sig_mean = second - (second - first) / 2
            if p_count >= 10:
                sig_mean = 0
            sig_means.append(sig_mean)
            # The below is to check things; turned off if running over lots of files.
            """
            plt.plot(bins[:-1], counter, c="blue", lw=2, label="Raw")
            plt.plot(bins[:-1], sgf_proton_3, c="red", lw=1, label="Smoothed")
            plt.plot(bins[:-1], sgf_proton_3_2, c="green", label="2nd derivative")
            for k, infl in enumerate(infls, 1):
                plt.axvline(x=infl, color='k', label=f'Inflection Point {k}')
            plt.axvline(x=sig_mean, c="pink", label="nSigmaMean")
            p_title = r'$p_T$ <= ' + str(0.1*(p_count+2))
            plt.title(p_title)
            plt.legend()
            plt.show()
            """
            p_count += 1
        sig_means = np.array(sig_means)

        # Now to modify nSigmaProton to be the difference between the values and
        # the found means.
        self.nsigma_proton = ak.where(self.p_t <= 0.1, self.nsigma_proton - sig_means[0], self.nsigma_proton)
        for k in range(1, len(sig_means)):
            self.nsigma_proton = ak.where((self.p_t > 0.1 * (k + 1)) & (self.p_t <= 0.1 * (k + 2)),
                                          self.nsigma_proton - sig_means[k], self.nsigma_proton)

    def calibrate_nsigmaproton_yu(self):
        """This is to calibrate nSigmaProton using Yu's Gaussian values."""
        slot = np.where(run_list == self.run_id)[0][0]
        calvals = sig[slot] * 1000  # The nSigmaProton calibrated values for our test arrays.
        # The rest wil be of the same form as the above function calibrate_nsigmaproton, but
        # obviously not calculating the values (just taking Yu's).
        deltas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
        for k in range(13):
            self.nsigma_proton = ak.where((self.p_g > deltas[k]) & (self.p_g <= deltas[k + 1]),
                                          self.nsigma_proton - calvals[k], self.nsigma_proton)

    def make_pandas(self, directory='nachos'):
        # Event dataframe
        self.event_df = pd.DataFrame(self.v_x, columns=['v_x'])
        self.event_df['v_y'] = self.v_y
        self.event_df['v_z'] = self.v_z
        self.event_df['v_r'] = self.v_r
        self.event_df['vz_vpd'] = self.vz_vpd
        self.event_df['refmult3'] = self.refmult3
        self.event_df['tofmult'] = self.tofmult
        self.event_df['tofmatch'] = self.tofmatch
        self.event_df['beta_eta_1'] = self.beta_eta_1
        self.event_df['zdcx'] = self.zdcx
        self.event_df['ring1'] = self.epd_hits[:, 0]
        self.event_df['ring2'] = self.epd_hits[:, 1]
        self.event_df['ring3'] = self.epd_hits[:, 2]
        self.event_df['ring4'] = self.epd_hits[:, 3]
        self.event_df['ring5'] = self.epd_hits[:, 4]
        self.event_df['ring6'] = self.epd_hits[:, 5]
        self.event_df['ring7'] = self.epd_hits[:, 6]
        self.event_df['ring8'] = self.epd_hits[:, 7]
        self.event_df['ring9'] = self.epd_hits[:, 8]
        self.event_df['ring10'] = self.epd_hits[:, 9]
        self.event_df['ring11'] = self.epd_hits[:, 10]
        self.event_df['ring12'] = self.epd_hits[:, 11]
        self.event_df['ring13'] = self.epd_hits[:, 12]
        self.event_df['ring14'] = self.epd_hits[:, 13]
        self.event_df['ring15'] = self.epd_hits[:, 14]
        self.event_df['ring16'] = self.epd_hits[:, 15]
        self.event_df['ring17'] = self.epd_hits[:, 16]
        self.event_df['ring18'] = self.epd_hits[:, 17]
        self.event_df['ring19'] = self.epd_hits[:, 18]
        self.event_df['ring20'] = self.epd_hits[:, 19]
        self.event_df['ring21'] = self.epd_hits[:, 20]
        self.event_df['ring22'] = self.epd_hits[:, 21]
        self.event_df['ring23'] = self.epd_hits[:, 22]
        self.event_df['ring24'] = self.epd_hits[:, 23]
        self.event_df['ring25'] = self.epd_hits[:, 24]
        self.event_df['ring26'] = self.epd_hits[:, 25]
        self.event_df['ring27'] = self.epd_hits[:, 26]
        self.event_df['ring28'] = self.epd_hits[:, 27]
        self.event_df['ring29'] = self.epd_hits[:, 28]
        self.event_df['ring30'] = self.epd_hits[:, 29]
        self.event_df['ring31'] = self.epd_hits[:, 30]
        self.event_df['ring32'] = self.epd_hits[:, 31]
        self.event_df['protons'] = self.protons_low + self.protons_high
        self.event_df['antiprotons'] = self.antiprotons_low + self.antiprotons_high
        self.event_df['net_protons'] = self.protons_low + self.protons_high - \
                                       self.antiprotons_low - self.antiprotons_high
        """
        # Track dataframe
        p_t_df = ak.to_pandas(self.p_t, anonymous='p_t')
        p_g_df = ak.to_pandas(self.p_g, anonymous='p_g')
        self.track_df = pd.concat((p_t_df, p_g_df), axis=1)
        phi_df = ak.to_pandas(self.phi, anonymous='phi')
        self.track_df = pd.concat((self.track_df, phi_df), axis=1)
        dca_df = ak.to_pandas(self.dca, anonymous='dca')
        self.track_df = pd.concat((self.track_df, dca_df), axis=1)
        eta_df = ak.to_pandas(self.eta, anonymous='eta')
        self.track_df = pd.concat((self.track_df, eta_df), axis=1)
        nhitsfit_df = ak.to_pandas(self.nhitsfit, anonymous='nhitsfit')
        self.track_df = pd.concat((self.track_df, nhitsfit_df), axis=1)
        nhitsdedx_df = ak.to_pandas(self.nhitsdedx, anonymous='nhitsdedx')
        self.track_df = pd.concat((self.track_df, nhitsdedx_df), axis=1)
        nhitsmax_df = ak.to_pandas(self.nhitsmax, anonymous='nhitsmax')
        self.track_df = pd.concat((self.track_df, nhitsmax_df), axis=1)
        dedx_df = ak.to_pandas(self.dedx, anonymous='dedx')
        self.track_df = pd.concat((self.track_df, dedx_df), axis=1)
        rapidity_df = ak.to_pandas(self.rapidity, anonymous='rapidity')
        self.track_df = pd.concat((self.track_df, rapidity_df), axis=1)
        nsigma_proton_df = ak.to_pandas(self.nsigma_proton, anonymous='nsigma_proton')
        self.track_df = pd.concat((self.track_df, nsigma_proton_df), axis=1)
        charge_df = ak.to_pandas(self.charge, anonymous='charge')
        self.track_df = pd.concat((self.track_df, charge_df), axis=1)

        # TOFTrack dataframe
        tofpid = ak.to_pandas(self.tofpid, anonymous='tofpid')
        mi = pd.MultiIndex.from_frame(tofpid.reset_index(level='subentry', drop=True).reset_index())
        self.toftrack_df = self.track_df.loc[mi.intersection(self.track_df.index)]
        beta = ak.to_pandas(self.beta, anonymous='beta')
        self.toftrack_df = pd.concat((self.toftrack_df, beta), axis=1)
        m_2 = ak.to_pandas(self.m_2, anonymous='m_2')
        self.toftrack_df = pd.concat((self.toftrack_df, m_2), axis=1)
        """
        if os.path.exists(directory + "\\protons.pkl"):
            temp = pd.read_pickle(directory + "\\protons.pkl")
            self.event_df = pd.concat((temp, self.event_df), axis=1)
        self.event_df.to_pickle(directory + "\\protons.pkl")

    def make_histos(self, directory='nachos', cut='event'):
        a, b, c, d = 1000, 161, 86, 101
        vz_count, vz_bins = np.histogram(self.v_z, bins=a, range=(-200, 200))
        vr_count, vr_binsX, vr_binsY = np.histogram2d(self.v_y, self.v_x, bins=a, range=((-10, 10), (-10, 10)))
        ref_count, ref_bins = np.histogram(self.refmult3, bins=a, range=(0, a))
        mpq_count, mpq_binsX, mpq_binsY = np.histogram2d(ak.to_numpy(ak.flatten(self.m_2)),
                                                          np.multiply(ak.to_numpy(ak.flatten(self.p_g_tof)),
                                                                      ak.to_numpy(ak.flatten(self.charge_tof))),
                                                          bins=a, range=((0, 1.5), (-5, 5)))
        rt_mult_count, rt_mult_binsX, rt_mult_binsY = np.histogram2d(self.tofmult, self.refmult3, bins=(1700, a),
                                                                     range=((0, 1700), (0, a)))
        rt_match_count, rt_match_binsX, rt_match_binsY = np.histogram2d(self.tofmatch, self.refmult3, bins=a,
                                                                        range=((0, 500), (0, a)))
        ref_beta_count, ref_beta_binsX, ref_beta_binsY = np.histogram2d(self.beta_eta_1, self.refmult3,
                                                                        bins=(400, a), range=((0, 400),
                                                                                              (0, a)))
        pt_count, pt_bins = np.histogram(ak.to_numpy(ak.flatten(self.p_t)), bins=a, range=(0, 6))
        phi_count, phi_bins = np.histogram(ak.to_numpy(ak.flatten(self.phi)), bins=a,
                                           range=(-np.pi - 0.2, np.pi + 0.2))
        dca_count, dca_bins = np.histogram(ak.to_numpy(ak.flatten(self.dca)), bins=a, range=(0, 5))
        eta_count, eta_bins = np.histogram(ak.to_numpy(ak.flatten(self.eta)), bins=a, range=(-3, 3))
        nhitsq_count, nhitsq_bins = np.histogram(ak.to_numpy(ak.flatten(self.nhitsfit)), bins=b,
                                                 range=(-(b - 1) / 2, (b - 1) / 2))
        nhits_dedx_count, nhits_dedx_bins = np.histogram(ak.to_numpy(ak.flatten(self.nhitsdedx)), bins=c,
                                                         range=(0, c - 1))
        betap_count, betap_binsX, betap_binsY = np.histogram2d(np.divide(1, ak.to_numpy(ak.flatten(self.beta))),
                                                               ak.to_numpy(ak.flatten(self.p_g_tof)),
                                                               bins=a, range=((0.5, 3.6), (0, 10)))
        dedx_pq_count, dedx_pq_binsX, dedx_pq_binsY = np.histogram2d(ak.to_numpy(ak.flatten(self.dedx)),
                                                                     np.multiply(ak.to_numpy(ak.flatten(self.charge)),
                                                                                 ak.to_numpy(ak.flatten(self.p_g))),
                                                                     bins=a, range=((0, 31), (-3, 3)))
        if os.path.exists(directory + "\\" + cut + "_vz_hist.npy"):
            temp = np.load(directory + "\\" + cut + "_vz_hist.npy",
                           allow_pickle=True)
            vz_count = np.add(temp[0], vz_count)
        np.save(directory + "\\" + cut + "_vz_hist.npy", (vz_count, vz_bins))
        if os.path.exists(directory + "\\" + cut + "_vr_hist.npy"):
            temp = np.load(directory + "\\" + cut + "_vr_hist.npy",
                           allow_pickle=True)
            vr_count = np.add(temp[0], vr_count)
        np.save(directory + "\\" + cut + "_vr_hist.npy", (vr_count, vr_binsX, vr_binsY))
        if os.path.exists(directory + "\\" + cut + "_ref_hist.npy"):
            temp = np.load(directory + "\\" + cut + "_ref_hist.npy",
                           allow_pickle=True)
            ref_count = np.add(temp[0], ref_count)
        np.save(directory + "\\" + cut + "_ref_hist.npy", (ref_count, ref_bins))
        if os.path.exists(directory + "\\" + cut + "_mpq_hist.npy"):
            temp = np.load(directory + "\\" + cut + "_mpq_hist.npy",
                           allow_pickle=True)
            mpq_count = np.add(temp[0], mpq_count)
        np.save(directory + "\\" + cut + "_mpq_hist.npy", (mpq_count, mpq_binsX, mpq_binsY))
        if os.path.exists(directory + "\\" + cut + "_rt_mult_hist.npy"):
            temp = np.load(directory + "\\" + cut + "_rt_mult_hist.npy",
                           allow_pickle=True)
            rt_mult_count = np.add(temp[0], rt_mult_count)
        np.save(directory + "\\" + cut + "_rt_mult_hist.npy", (rt_mult_count, rt_mult_binsX, rt_mult_binsY))
        if os.path.exists(directory + "\\" + cut + "_rt_match_hist.npy"):
            temp = np.load(directory + "\\" + cut + "_rt_match_hist.npy",
                           allow_pickle=True)
            rt_match_count = np.add(temp[0], rt_match_count)
        np.save(directory + "\\" + cut + "_rt_match_hist.npy", (rt_match_count, rt_match_binsX, rt_match_binsY))
        if os.path.exists(directory + "\\" + cut + "_ref_beta_hist.npy"):
            temp = np.load(directory + "\\" + cut + "_ref_beta_hist.npy",
                           allow_pickle=True)
            ref_beta_count = np.add(temp[0], ref_beta_count)
        np.save(directory + "\\" + cut + "_ref_beta_hist.npy", (ref_beta_count, ref_beta_binsX, ref_beta_binsY))
        if os.path.exists(directory + "\\" + cut + "_pt_hist.npy"):
            temp = np.load(directory + "\\" + cut + "_pt_hist.npy",
                           allow_pickle=True)
            pt_count = np.add(temp[0], pt_count)
        np.save(directory + "\\" + cut + "_pt_hist.npy", (pt_count, pt_bins))
        if os.path.exists(directory + "\\" + cut + "_phi_hist.npy"):
            temp = np.load(directory + "\\" + cut + "_phi_hist.npy",
                           allow_pickle=True)
            phi_count = np.add(temp[0], phi_count)
        np.save(directory + "\\" + cut + "_phi_hist.npy", (phi_count, phi_bins))
        if os.path.exists(directory + "\\" + cut + "_dca_hist.npy"):
            temp = np.load(directory + "\\" + cut + "_dca_hist.npy",
                           allow_pickle=True)
            dca_count = np.add(temp[0], dca_count)
        np.save(directory + "\\" + cut + "_dca_hist.npy", (dca_count, dca_bins))
        if os.path.exists(directory + "\\" + cut + "_eta_hist.npy"):
            temp = np.load(directory + "\\" + cut + "_eta_hist.npy",
                           allow_pickle=True)
            eta_count = np.add(temp[0], eta_count)
        np.save(directory + "\\" + cut + "_eta_hist.npy", (eta_count, eta_bins))
        if os.path.exists(directory + "\\" + cut + "_nhitsq_hist.npy"):
            temp = np.load(directory + "\\" + cut + "_nhitsq_hist.npy",
                           allow_pickle=True)
            nhitsq_count = np.add(temp[0], nhitsq_count)
        np.save(directory + "\\" + cut + "_nhitsq_hist.npy", (nhitsq_count, nhitsq_bins))
        if os.path.exists(directory + "\\" + cut + "_nhits_dedx_hist.npy"):
            temp = np.load(directory + "\\" + cut + "_nhits_dedx_hist.npy",
                           allow_pickle=True)
            nhits_dedx_count = np.add(temp[0], nhits_dedx_count)
        np.save(directory + "\\" + cut + "_nhits_dedx_hist.npy", (nhits_dedx_count, nhits_dedx_bins))
        if os.path.exists(directory + "\\" + cut + "_betap_hist.npy"):
            temp = np.load(directory + "\\" + cut + "_betap_hist.npy",
                           allow_pickle=True)
            betap_count = np.add(temp[0], betap_count)
        np.save(directory + "\\" + cut + "_betap_hist.npy", (betap_count, betap_binsX, betap_binsY))
        if os.path.exists(directory + "\\" + cut + "_dedx_pq_hist.npy"):
            temp = np.load(directory + "\\" + cut + "_dedx_pq_hist.npy",
                           allow_pickle=True)
            dedx_pq_count = np.add(temp[0], dedx_pq_count)
        np.save(directory + "\\" + cut + "_dedx_pq_hist.npy", (dedx_pq_count, dedx_pq_binsX, dedx_pq_binsY))

    def make_aves(self, directory='nachos', cut='none'):
        if cut == 'none':
            r = np.sqrt(len(self.refmult3))
            data = {'refmult3_ave': [np.mean(self.refmult3)], 'refmult3_std': [np.std(self.refmult3)]/r,
                    'v_z_ave': [np.mean(self.v_z)], 'v_z_std': [np.std(self.v_z)]/r,
                    'p_t_ave': [np.mean(self.p_t)], 'p_t_std': [np.std(self.p_t)]/r,
                    'phi_ave': [np.mean(self.phi)], 'phi_std': [np.std(self.phi)]/r,
                    'v_r_ave': [np.mean(self.v_r)], 'v_r_std': [np.std(self.v_r)]/r,
                    'zdcx_ave': [np.mean(self.zdcx)], 'zdcx_std': [np.std(self.zdcx)]/r,
                    'eta_ave': [np.mean(self.eta)], 'eta_std': [np.std(self.eta)]/r,
                    'dca_ave': [np.mean(self.dca)], 'dca_std': [np.std(self.dca)]/r}
            self.ave_df = pd.DataFrame(data, index=[self.run_id])
        if cut == 'pid':
            r = np.sqrt(len(self.protons_low))
            self.ave_df['protons_low_ave'] = np.mean(self.protons_low)
            self.ave_df['protons_low_std'] = np.std(self.protons_low)/r
            self.ave_df['antiprotons_low_ave'] = np.mean(self.antiprotons_low)
            self.ave_df['antiprotons_low_std'] = np.std(self.antiprotons_low)/r
            self.ave_df['protons_high_ave'] = np.mean(self.protons_high)
            self.ave_df['protons_high_std'] = np.std(self.protons_high)/r
            self.ave_df['antiprotons_high_ave'] = np.mean(self.antiprotons_high)
            self.ave_df['antiprotons_high_std'] = np.std(self.antiprotons_high)/r
            if os.path.exists(directory + "\\" + "averages.pkl"):
                df = pd.read_pickle(directory + "\\" + "averages.pkl")
                self.ave_df = pd.concat((df, self.ave_df))
            self.ave_df.to_pickle(directory + "\\" + "averages.pkl")


class EventCuts:
    """
    Event_Cuts takes an object of events and applys the criteria function to it
    to determin acceptable events.  They can then be accessed in the same way as
    a typical Pico_DST object.
    The criteria function must accept a object of events and the event index as the parameters.
    A typical function might be:

    def basic_vertex_filter(events):
        select = events.v_r <= 2.0  # All events within 2.0 cm of the origin in the transverse plane
        select = select & np.abs(events.v_z) <= 30.0  # Additional parameter of being within 30.0 cm in z
        return select
    """

    def __init__(self, events, criteria=None, mask=None):
        self.events = events
        self.mask = mask  # Sometimes we want to use the same mask
        if self.mask is None:
            self.mask = self.generate_mask(criteria)
        # self.num_events = int(np.sum(self.mask))  # <---- CULPRIT!!!
        # print(self.mask)

    def generate_mask(self, criteria, mask=None):
        """
        generate_mask applies the criteria function to each event to determine if it is acceptable
        """
        if mask is None:
            mask = criteria
        else:
            mask = mask * criteria
        return mask

    def __getattr__(self, name):
        if name == "epd_hits":  # Create an new event cut object for EPD data
            return EventCuts(self.events.epd_hits, mask=self.mask)
        array = getattr(self.events, name)[self.mask]  # Return a filterd array, a new copy
        return array


def read_pico(data, directory):
    if data != 'taco':
        try:
            bad_runs = np.loadtxt(r"C:\Users\dansk\Documents\Thesis\Protons\2021_Analysis_Python\bad_runs.txt")
            p_pico = PicoDST()
            p_pico.import_data(data)
            if p_pico.run_id in bad_runs:
                print("Skipped run", p_pico.run_id, "due to run QA.")
                return
            p_pico.calibrate_nsigmaproton_yu()
            # p_pico.make_histos(directory=directory, cut='raw')
            # p_pico.make_aves(directory=directory, cut='none')
            p_pico.event_cuts()
            p_pico.track_qa_cuts()
            # p_pico.make_histos(directory=directory, cut='qa')
            p_pico.select_protons()
            # p_pico.make_aves(directory=directory, cut='pid')
            p_pico.make_pandas(directory=directory)
            print(p_pico.run_id, "processed.")
        except Exception as exc:  # For any issues that might pop up.
            print("Error on ", data)
            print(exc)
    else:
        return
