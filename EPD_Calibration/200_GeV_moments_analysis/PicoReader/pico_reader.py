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
from scipy.stats import skew, kurtosis, moment
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time

# Speed of light, in m/s
SPEED_OF_LIGHT = 299792458
# Proton mass, in GeV
PROTON_MASS = 0.9382720813

run_list = np.loadtxt(r'C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\runs.txt', delimiter=',').astype("int")
sig = np.loadtxt(r'C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\nsigmaVals.txt', delimiter=',')  # [[447, 669]]


def ak_to_numpy_flat(*args):
    """This transforms an awkward array into a flat, numpy array for plotting.
        Args:
            *args: The array(s) to flatten and go numpy."""
    for arg in args:
        arg = ak.to_numpy(ak.flatten(arg))
        yield arg


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


def get_ave(arr):
    """This gives the mean and standard error for a numpy array.
        Args:
            arr: Array you want to get the mean and std error from"""
    ave = np.mean(arr)
    dev = np.divide(np.std(arr), len(arr))
    return ave, dev


# TODO Check to see that these results are reasonable. Graph against p_t.
def rapidity(p_z, p_g):
    e_p = np.sqrt(np.add(PROTON_MASS ** 2, np.power(p_g, 2)))
    y = np.multiply(0.5, np.log(np.divide(np.add(e_p, p_z), np.subtract(e_p, p_z))))
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
    taco = None

    def __init__(self, mID, mQT_data, mnMips, cal_file=None, lower_bound=0.2, upper_bound=3):
        self.mID = mID
        self.mQT_data = mQT_data
        self.mnMip = mnMips

        self.has_TAC = np.bitwise_and(np.right_shift(self.mQT_data, 29), 0x1)
        self.status_is_good = np.bitwise_and(np.right_shift(self.mQT_data, 30), 0x1)

        self.adc = np.bitwise_and(self.mQT_data, 0x0FFF)
        self.tac = np.bitwise_and(np.right_shift(self.mQT_data, 12), 0x0FFF)
        self.TDC = np.bitwise_and(np.right_shift(self.mQT_data, 24), 0x001F)

        self.EW = ak.Array(np.sign(self.mID))
        self.position = np.abs(self.mID) // 100
        self.tiles = np.abs(self.mID) % 100
        self.row = ((np.abs(self.mID) % 100) // 2) + 1
        # The following is to use calibrated values for FastOffline.
        # This is for using the .txt method.
        """
        if cal_file is not None:
            ew = ak.where(self.mID > 0, 1, 0)
            pp = self.position - 1
            tt = self.tiles - 1
            epd_flat = (ew * 372) + (pp * 31) + tt
            counts = ak.num(epd_flat)
            array = ak.flatten(epd_flat)
            cal_vals = cal_file[array]
            cal_vals = ak.unflatten(cal_vals, counts)
            self.nMip = ak.where(self.status_is_good, self.adc / cal_vals, 0)
        """
        # And for if using the numpy file ([ew][pp-1][tt-1]).
        if cal_file is not None:
            ew = ak.where(self.mID > 0, 1, 0)
            pp = self.position - 1
            tt = self.tiles - 1
            epd_flat = (ew * 372) + (pp * 31) + tt
            counts = ak.num(epd_flat)
            array = ak.flatten(epd_flat)
            cal_file = cal_file.flatten()
            cal_vals = cal_file[array]
            cal_vals = ak.unflatten(cal_vals, counts)
            self.nMip = ak.where(self.status_is_good, self.adc / cal_vals, 0)
        else:
            self.nMip = ak.where(self.status_is_good, self.mnMip, 0)
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
        return ring_sum


class PicoDST:
    """This class makes the PicoDST from the root file, along with
    all the observables used for proton kurtosis analysis."""

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
        self.p_t = None
        self.p_g = None
        self.phi = None
        self.dca = None
        self.eta = None
        self.nhitsfit = None
        self.nhitsdedx = None
        self.dedx = None
        self.zdcx = None
        self.rapidity = None
        self.nhitsmax = None
        self.nsigma_proton = None
        self.nsigma_pion = None
        self.nsigma_electron = None
        self.nsigma_kaon = None
        self.tofpid = None
        self.m_2 = None
        self.charge = None
        self.beta = None
        self.protons = None
        self.antiprotons = None
        self.dedx_histo = None
        self.p_g_histo = None
        self.charge_histo = None
        self.run_id = None
        self.event_df = None
        self.track_df = None
        self.toftrack_df = None
        self.time_blocks = []
        self.block_2 = []
        self.block_3 = []
        self.block_EPD = []

        if data_file is not None:
            self.import_data(data_file)

    def import_data(self, data_in, cal_file=None, nsig_cals=None):
        """This imports the data. You must have the latest versions
        of uproot and awkward installed on your machine (uproot4 and
        awkward 1.0 as of the time of this writing).
        Use pip install uproot awkward.
        Args:
            data_in (str): The path to the picoDst ROOT file
            cal_file (file): Calibration file for EPD (None by default)
            nsig_cals (float): Calibration numbers for nSigma values"""
        try:
            with up.open(data_in) as data:
                t_0 = time.time()
                ################### Time Block 1 ################################################
                self.num_events = len(data["PicoDst"]["Event"]["Event.mPrimaryVertexX"].array())
                self.run_id = int(ak.to_numpy(ak.flatten(data["PicoDst"]["Event"]["Event.mRunId"].array()))[0])
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
                t_1 = time.time()
                self.time_blocks.append(t_1-t_0)
                ################### Time Block 2 ################################################
                # Make p_g and p_t
                p_x = data["PicoDst"]["Track"]["Track.mGMomentumX"].array()
                p_y = data["PicoDst"]["Track"]["Track.mGMomentumY"].array()
                # p_y = ak.where(p_y == 0.0, 1e-10, p_y)  # to avoid infinities
                p_z = data["PicoDst"]["Track"]["Track.mGMomentumZ"].array()

                # This is a little faster. To optimise, do this with all track arrays.
                """
                counts = ak.num(p_x)
                p_x = ak.flatten(p_x)
                p_y = ak.flatten(p_y)
                p_z = ak.flatten(p_z)
                p_t = np.sqrt(np.add(np.power(p_x, 2), np.power(p_y, 2)))
                p_g = np.sqrt(np.power(p_x, 2) + np.power(p_y, 2) + np.power(p_z, 2))
                eta = np.arcsinh(np.divide(p_z, p_t))
                rapid = rapidity(p_z, p_g)
                self.p_t = ak.unflatten(p_t, counts)
                self.p_g = ak.unflatten(p_g, counts)
                self.eta = ak.unflatten(eta, counts)
                self.rapidity = ak.unflatten(rapid, counts)
                p_x = ak.unflatten(p_x, counts)
                p_y = ak.unflatten(p_y, counts)
                p_z = ak.unflatten(p_z, counts)

                """
                delta_0 = time.time()
                self.p_t = np.sqrt(np.add(np.power(p_x, 2), np.power(p_y, 2)))
                delta_1 = time.time()
                self.p_g = np.sqrt(np.power(p_x, 2) + np.power(p_y, 2) + np.power(p_z, 2))
                delta_2 = time.time()
                self.eta = np.arcsinh(np.divide(p_z, self.p_t))
                delta_3 = time.time()
                self.rapidity = rapidity(p_z, self.p_g)
                t_2 = time.time()
                self.time_blocks.append(t_2-t_1)
                self.block_2.append((delta_1-delta_0, delta_2-delta_1, delta_3-delta_2, t_2-delta_3))
                ################### Time Block 3 ################################################
                # Make dca
                o_x = data["PicoDst"]["Track"]["Track.mOriginX"].array()
                o_y = data["PicoDst"]["Track"]["Track.mOriginY"].array()
                o_z = data["PicoDst"]["Track"]["Track.mOriginZ"].array()
                delta_1 = time.time()
                self.dca = np.sqrt((np.power(o_x, 2) + np.power(o_y, 2) + np.power(o_z, 2))) / 100.0
                self.nhitsdedx = data["PicoDst"]["Track"]["Track.mNHitsDedx"].array()
                self.nhitsfit = data["PicoDst"]["Track"]["Track.mNHitsFit"].array()
                self.nhitsmax = data["PicoDst"]["Track"]["Track.mNHitsMax"].array()
                delta_2 = time.time()
                self.nhitsmax = ak.where(self.nhitsmax == 0, 1e-10, self.nhitsmax)  # to avoid infinities
                delta_3 = time.time()
                self.dedx = data["PicoDst"]["Track"]["Track.mDedx"].array()
                self.nsigma_proton = data["PicoDst"]["Track"]["Track.mNSigmaProton"].array() / 1000.0
                self.nsigma_pion = data["PicoDst"]["Track"]["Track.mNSigmaPion"].array() / 1000.0
                self.nsigma_electron = data["PicoDst"]["Track"]["Track.mNSigmaElectron"].array() / 1000.0
                self.nsigma_kaon = data["PicoDst"]["Track"]["Track.mNSigmaKaon"].array() / 1000.0
                t_3 = time.time()
                self.time_blocks.append(t_3-t_2)
                self.block_3.append((delta_1-t_2, delta_2-delta_1, delta_3-delta_2, t_3-delta_3))
                ################### Time Block 4 ################################################
                if nsig_cals is not None:
                    for j in range(11):
                        self.nsigma_proton = ak.where((self.p_g > 0.1 * j + 0.1) &
                                                      (self.p_g <= 0.1 * j + 0.2),
                                                      self.nsigma_proton - nsig_cals[j],
                                                      self.nsigma_proton)
                self.charge = ak.where(self.nhitsfit >= 0, 1, -1)
                self.beta = data["PicoDst"]["BTofPidTraits"]["BTofPidTraits.mBTofBeta"].array() / 20000.0
                self.tofpid = data["PicoDst"]["BTofPidTraits"]["BTofPidTraits.mTrackIndex"].array()
                # Make B_n_1
                be1_1 = ak.where(self.beta > 0.1, 1, 0)
                be1_2 = ak.where(np.absolute(self.eta[self.tofpid]) < 1.0, 1, 0)
                be1_3 = ak.where(self.dca[self.tofpid] < 3.0, 1, 0)
                be1_4 = ak.where(np.absolute(self.nhitsfit[self.tofpid]) > 10, 1, 0)
                be1 = be1_1 * be1_2 * be1_3 * be1_4
                self.beta_eta_1 = ak.sum(be1, axis=-1)
                t_4 = time.time()
                self.time_blocks.append(t_4-t_3)
                ################### Time Block 5 ################################################
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
                self.tofmatch = ak.sum(nTofMatchTot, axis=-1)
                # Make m^2
                p_squared = np.power(self.p_g[self.tofpid], 2)
                b_squared = ak.where(self.beta == 0.0, 1e-10, self.beta)  # to avoid infinities
                b_squared = np.divide(1, np.power(b_squared, 2)) - 1
                self.m_2 = np.multiply(p_squared, b_squared)
                # Make phi.
                self.phi = np.arctan2(p_y, p_x)
                t_5 = time.time()
                self.time_blocks.append(t_5-t_4)
                ################### Time Block 6 ################################################
                # Load EPD Data
                epd_hit_id_data = data["PicoDst"]["EpdHit"]["EpdHit.mId"].array()
                epd_hit_mQTdata = data["PicoDst"]["EpdHit"]["EpdHit.mQTdata"].array()
                epd_hit_mnMIP = data["PicoDst"]["EpdHit"]["EpdHit.mnMIP"].array()
                delta_1 = time.time()
                self.epd_hits = EPD_Hits(epd_hit_id_data, epd_hit_mQTdata, epd_hit_mnMIP, cal_file)
                delta_2 = time.time()
                self.epd_hits = self.epd_hits.generate_epd_hit_matrix()
                t_6 = time.time()
                self.time_blocks.append(t_6-t_5)
                self.block_EPD.append((delta_1-t_5, delta_2-delta_1, t_6-delta_2))
                ################################################################################

        # except ValueError:  # Skip empty picos.
        #     print("ValueError at: " + data_in)  # Identifies the misbehaving file.
        except KeyError:  # Skip non-empty picos that have no data.
            print("KeyError at: " + data_in)  # Identifies the misbehaving file.

    def event_cuts(self, v_r_cut=2.0, v_z_cut=30.0, v_cut=1.0e-5,
                   tofmult_refmult=np.array([[2.536, 200], [1.352, -54.08]]),
                   tofmatch_refmult=np.array([0.239, -14.34]), beta_refmult=np.array([0.447, -17.88])):
        index = ((np.absolute(self.v_z) <= v_z_cut) & (self.v_r < v_r_cut) &
                 (self.tofmult <= (np.multiply(tofmult_refmult[0][0], self.refmult3) + tofmult_refmult[0][1])) &
                 (self.tofmult >= (np.multiply(tofmult_refmult[1][0], self.refmult3) + tofmult_refmult[1][1])) &
                 (self.tofmatch >= (np.multiply(tofmatch_refmult[0], self.refmult3) + tofmatch_refmult[1])) &
                 (self.beta_eta_1 >= (np.multiply(beta_refmult[0], self.refmult3) + beta_refmult[1])) &
                 (abs(self.v_x) >= v_cut) & (abs(self.v_y) >= v_cut) & (abs(self.v_z) >= v_cut))
        self.v_x, self.v_y, self.v_z, self.v_r, self.refmult3, self.tofmult, \
            self.tofmatch, self.beta_eta_1, self.p_t, self.p_g, self.phi, self.dca, \
            self.eta, self.nhitsfit, self.nhitsdedx, self.m_2, self.charge, self.beta, \
            self.dedx, self.zdcx, self.rapidity, self.nhitsmax, self.nsigma_proton, \
            self.tofpid, self.vz_vpd = \
            index_cut(index, self.v_x, self.v_y, self.v_z, self.v_r, self.refmult3,
                      self.tofmult, self.tofmatch, self.beta_eta_1, self.p_t, self.p_g,
                      self.phi, self.dca, self.eta, self.nhitsfit, self.nhitsdedx,
                      self.m_2, self.charge, self.beta, self.dedx, self.zdcx,
                      self.rapidity, self.nhitsmax, self.nsigma_proton, self.tofpid,
                      self.vz_vpd)
        new_epd = []
        for i in range(32):
            new_epd.append(self.epd_hits[i][index])
        new_epd = ak.Array(new_epd)
        self.epd_hits = new_epd

    def track_qa_cuts(self, nhitsdedx_cut=5, nhitsfit_cut=20, ratio=0.52, dca_cut=1.0,
                      pt_low_cut=0.4, pt_high_cut=2.0, rapid_cut=0.5):
        index = ((self.nhitsdedx > nhitsdedx_cut) & (np.absolute(self.nhitsfit) > nhitsfit_cut) &
                     (np.divide(1 + np.absolute(self.nhitsfit), 1 + self.nhitsmax) > ratio) &
                 (self.dca < dca_cut) & (self.p_t > pt_low_cut) &
                 (self.p_t < pt_high_cut) & (np.absolute(self.rapidity) <= rapid_cut))
        self.p_t, self.p_g, self.phi, self.dca, self.eta, self.nhitsfit, self.nhitsdedx, self.charge, self.dedx, \
            self.rapidity, self.nhitsmax, self.nsigma_proton = \
            index_cut(index, self.p_t, self.p_g, self.phi, self.dca, self.eta, self.nhitsfit,
                      self.nhitsdedx, self.charge, self.dedx, self.rapidity, self.nhitsmax,
                      self.nsigma_proton)

    def track_qa_cuts_tof(self, nhitsdedx_cut=5, nhitsfit_cut=20, ratio=0.52, dca_cut=1.0,
                          pt_low_cut=0.4, pt_high_cut=2.0, rapid_cut=0.5):
        self.p_t, self.p_g, self.phi, self.dca, self.eta, self.nhitsfit, self.nhitsdedx,\
            self.charge, self.dedx, self.rapidity, self.nhitsmax, self.nsigma_proton = \
            index_cut(self.tofpid, self.p_t, self.p_g, self.phi, self.dca, self.eta,
                      self.nhitsfit, self.nhitsdedx, self.charge, self.dedx, self.rapidity,
                      self.nhitsmax, self.nsigma_proton)
        index = ((self.nhitsdedx > nhitsdedx_cut) & (np.absolute(self.nhitsfit) > nhitsfit_cut) &
                 (np.divide(1 + np.absolute(self.nhitsfit), 1 + self.nhitsmax) > ratio) &
                 (self.dca < dca_cut) & (self.p_t > pt_low_cut) &
                 (self.p_t < pt_high_cut) & (np.absolute(self.rapidity) <= rapid_cut))
        self.p_t, self.p_g, self.phi, self.dca, self.eta, self.nhitsfit, self.nhitsdedx,\
            self.charge, self.dedx, self.rapidity, self.nhitsmax, self.nsigma_proton,\
            self.beta, self.m_2 = \
            index_cut(index, self.p_t, self.p_g, self.phi, self.dca, self.eta, self.nhitsfit,
                      self.nhitsdedx, self.charge, self.dedx, self.rapidity, self.nhitsmax,
                      self.nsigma_proton, self.beta, self.m_2)

    def select_protons_low(self, nsigma_cut=2.0, pt_high_cut=0.8, pg_cut=1.0,
                           mass_low_cut=0.6, mass_high_cut=1.2):
        index = ((np.abs(self.nsigma_proton) <= nsigma_cut) & (self.p_t < pt_high_cut) &
                 (self.p_g <= pg_cut))
        self.p_t, self.p_g, self.phi, self.dca, self.eta, self.nhitsfit, self.nhitsdedx,\
            self.charge, self.dedx, self.rapidity, self.nhitsmax, self.nsigma_proton = \
            index_cut(index, self.p_t, self.p_g, self.phi, self.dca, self.eta, self.nhitsfit,
                      self.nhitsdedx, self.charge, self.dedx, self.rapidity, self.nhitsmax,
                      self.nsigma_proton)
        protons = ak.where(self.charge > 0, 1, 0)
        antiprotons = ak.where(self.charge < 0, 1, 0)
        self.protons = ak.sum(protons, axis=-1)
        self.antiprotons = ak.sum(antiprotons, axis=-1)

    def select_protons_high(self, nsigma_cut=2.0, pt_low_cut=0.8, pg_cut=3.0,
                            mass_low_cut=0.6, mass_high_cut=1.2):
        index = ((np.abs(self.nsigma_proton) <= nsigma_cut) & (self.p_t >= pt_low_cut) &
                 (self.p_g <= pg_cut) & (self.m_2 >= mass_low_cut) & (self.m_2 <= mass_high_cut))
        self.p_t, self.p_g, self.phi, self.dca, self.eta, self.nhitsfit, self.nhitsdedx, self.beta, \
            self.charge, self.dedx, self.rapidity, self.nhitsmax, self.nsigma_proton, self.m_2 = \
            index_cut(index, self.p_t, self.p_g, self.phi, self.dca, self.eta, self.nhitsfit,
                      self.nhitsdedx, self.beta, self.charge, self.dedx, self.rapidity, self.nhitsmax,
                      self.nsigma_proton, self.m_2)
        protons = ak.where(self.charge > 0, 1, 0)
        antiprotons = ak.where(self.charge < 0, 1, 0)
        self.protons = ak.sum(protons, axis=-1)
        self.antiprotons = ak.sum(antiprotons, axis=-1)

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

    def make_pandas(self):
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
        self.event_df['ring1'] = self.epd_hits[0]
        self.event_df['ring2'] = self.epd_hits[1]
        self.event_df['ring3'] = self.epd_hits[2]
        self.event_df['ring4'] = self.epd_hits[3]
        self.event_df['ring5'] = self.epd_hits[4]
        self.event_df['ring6'] = self.epd_hits[5]
        self.event_df['ring7'] = self.epd_hits[6]
        self.event_df['ring8'] = self.epd_hits[7]
        self.event_df['ring9'] = self.epd_hits[8]
        self.event_df['ring10'] = self.epd_hits[9]
        self.event_df['ring11'] = self.epd_hits[10]
        self.event_df['ring12'] = self.epd_hits[11]
        self.event_df['ring13'] = self.epd_hits[12]
        self.event_df['ring14'] = self.epd_hits[13]
        self.event_df['ring15'] = self.epd_hits[14]
        self.event_df['ring16'] = self.epd_hits[15]
        self.event_df['ring17'] = self.epd_hits[16]
        self.event_df['ring18'] = self.epd_hits[17]
        self.event_df['ring19'] = self.epd_hits[18]
        self.event_df['ring20'] = self.epd_hits[19]
        self.event_df['ring21'] = self.epd_hits[20]
        self.event_df['ring22'] = self.epd_hits[21]
        self.event_df['ring23'] = self.epd_hits[22]
        self.event_df['ring24'] = self.epd_hits[23]
        self.event_df['ring25'] = self.epd_hits[24]
        self.event_df['ring26'] = self.epd_hits[25]
        self.event_df['ring27'] = self.epd_hits[26]
        self.event_df['ring28'] = self.epd_hits[27]
        self.event_df['ring29'] = self.epd_hits[28]
        self.event_df['ring30'] = self.epd_hits[29]
        self.event_df['ring31'] = self.epd_hits[30]
        self.event_df['ring32'] = self.epd_hits[31]

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


def moment_arr(arr):
    u = [np.mean(arr)]
    for i in range(2, 9):
        u.append(moment(arr, i))
    return u


def err(arr, u):
    n = np.sqrt(len(arr))
    e0 = np.sqrt(u[1])
    e1 = np.sqrt(u[3] - (u[1] ** 2))
    e2 = 9 * (u[1] ** 3) - 6 * (u[1] * u[3]) - (u[2] ** 2) + u[5]
    e3 = u[7] - 36 * (u[1] ** 4) + 48 * (u[1] * u[3]) + 64 * (u[1] * (u[2] ** 2)) - 12 * (u[1] * u[5]) - 8 * (
            u[2] * u[4]) - (u[3] ** 2)
    e = (e0, e1, e2, e3)
    return e


def err_rat(arr, u):
    e_r0 = (u[3] - (u[1] ** 2)) / (u[0] ** 2) - (2 * u[1] * u[2]) / (u[0] ** 3) + ((u[1] ** 3) / (u[0] ** 4))
    if np.isnan(e_r0):
        e_r0 = 0
    e_r1 = (9 * u[1] ** 2 - (6 * u[3]) / u[1] + (6 * u[2] ** 2 + u[5]) / u[1] ** 2 -
            (2 * u[4] * u[2]) / u[1] ** 3 + ((u[2] ** 2) * u[3]) / u[1] ** 4)
    if np.isnan(e_r1):
        e_r1 = 0
    e_r2 = (9 * (u[3] - u[1] ** 2) + (40 * (u[2] ** 2) - 6 * u[5]) / u[1] +
            (u[7] + 6 * (u[3] ** 2) - 8 * u[2] * u[4]) / u[1] ** 2 +
            (8 * u[3] * (u[2] ** 4) - 2 * u[3] * u[5]) / u[1] ** 3 +
            (u[3] ** 3) / u[1] ** 4)
    if np.isnan(e_r2):
        e_r2 = 0
    e_r = (e_r0, e_r1, e_r2)
    return e_r


def cbwc(df):
    pass
