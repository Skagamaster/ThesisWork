import numpy as np
import uproot as up
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

"""
a, b, c, d = 1000, 161, 86, 101
vz_count, vz_bins = np.histogram(0, bins=a, range=(-200, 200))
vr_count, vr_binsX, vr_binsY = np.histogram2d([0], [0], bins=a, range=((-10, 10), (-10, 10)))
ref_count, ref_bins = np.histogram(0, bins=a, range=(0, a))
mpq_count, mpq_binsX, mpq_binsY = np.histogram2d([0], [0], bins=a, range=((0, 1.5), (-5, 5)))
rt_mult_count, rt_mult_binsX, rt_mult_binsY = np.histogram2d([0], [0], bins=(1700, a), range=((0, 1700), (0, 1000)))
rt_match_count, rt_match_binsX, rt_match_binsY = np.histogram2d([0], [0], bins=a, range=((0, 500), (0, 1000)))
ref_beta_count, ref_beta_binsX, ref_beta_binsY = np.histogram2d([0], [0], bins=(400, a), range=((0, 400), (0, 1000)))
pt_count, pt_bins = np.histogram(0, bins=a, range=(0, 6))
phi_count, phi_bins = np.histogram(0, bins=a, range=(-np.pi - 0.2, np.pi + 0.2))
dca_count, dca_bins = np.histogram(0, bins=a, range=(0, 5))
eta_count, eta_bins = np.histogram(0, bins=a, range=(-3, 3))
nhitsq_count, nhitsq_bins = np.histogram(0, bins=b, range=(-(b-1)/2, (b-1)/2))
nhits_dedx_count, nhits_dedx_bins = np.histogram(0, bins=c, range=(0, c-1))
betap_count, betap_binsX, betap_binsY = np.histogram2d([0], [0], bins=a, range=((0.5, 3.6), (0, 10)))
dedx_pq_count, dedx_pq_binsX, dedx_pq_binsY = np.histogram2d([0], [0], bins=a, range=((0, 31), (-3, 3)))
"""


class HistoData:
    def __init__(self, data_file=None):
        self.v_z = None
        self.v_z_X = np.linspace(-200, 200, 1000)
        self.v_r = None
        self.v_r_X, self.v_r_Y = np.meshgrid(np.linspace(-10, 10, 1000), np.linspace(-10, 10, 1000))
        self.refmult = None
        self.refmultX = np.linspace(0, 999, 1000)
        self.rt_mult = None
        self.rtX, self.rtY = np.meshgrid(np.linspace(0, 999, 1000), np.linspace(0, 1699, 1700))
        self.rt_match = None
        self.rtmX, self.rtmY = np.meshgrid(np.linspace(0, 999, 1000), np.linspace(0, 500, 1000))
        self.ref_beta = None
        self.r_bX, self.r_bY = np.meshgrid(np.linspace(0, 999, 1000), np.linspace(0, 399, 400))
        self.mpq = None
        self.mpX, self.mpY = np.meshgrid(np.linspace(-5, 5, 1000), np.linspace(0, 1.5, 1000))
        self.betap = None
        self.beta_p_X, self.beta_p_Y = np.meshgrid(np.linspace(0.5, 3.6, 1000), np.linspace(0, 10, 1000))
        self.p_t = None
        self.phi = None
        self.dca = None
        self.eta = None
        self.rap = None
        self.nhitsq = None
        self.nhits_dedx = None
        self.nhitsfit_ratio = None
        self.dedx_pq = None
        self.dedxX, self.dedxY = np.meshgrid(np.linspace(-3, 3, 1000), np.linspace(0, 31, 1000))
        self.av_z = None
        self.av_r = None
        self.arefmult = None
        self.art_mult = None
        self.art_match = None
        self.aref_beta = None
        self.ampq = None
        self.abetap = None
        self.ap_t = None
        self.aphi = None
        self.adca = None
        self.aeta = None
        self.arap = None
        self.anhitsq = None
        self.anhits_dedx = None
        self.anhitsfit_ratio = None
        self.adedx_pq = None
        self.pdedx_pq = None
        self.p_t_p_g = None
        self.msq = None
        self.hNSigmaProton_0 = None
        self.hNSigmaProton_1 = None
        self.hNSigmaProton_2 = None
        self.hNSigmaProton_3 = None
        self.hNSigmaProton_4 = None
        self.hNSigmaProton_5 = None
        self.hNSigmaProton_6 = None
        self.hNSigmaProton_7 = None
        self.hNSigmaProton_8 = None
        self.hNSigmaProton_9 = None
        self.hNSigmaProton_10 = None
        self.Protons = None

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
                # Turned off for now as I'll just run over the full set.
                # file_num = data_in.rpartition('.')[0][4:]
                self.v_z = data['v_z'].values()
                self.v_r = data['v_r'].values()
                self.refmult = data['refmult'].values()
                self.rt_mult = data['rt_mult'].values()
                self.rt_match = data['rt_matchd'].values()
                self.ref_beta = data['ref_beta'].values()
                self.mpq = data['mpq'].values()
                self.betap = data['betapd'].values()
                self.p_t = data['p_t'].values()
                self.phi = data['phi'].values()
                self.dca = data['dca'].values()
                self.eta = data['eta'].values()
                self.rap = data['rap'].values()
                self.nhitsq = data['nhitsq'].values()
                self.nhits_dedx = data['nhits_dedx'].values()
                self.nhitsfit_ratio = data['nhitsfit_ratio'].values()
                self.dedx_pq = data['dedx_pq'].values()
                self.av_z = data['av_z'].values()
                self.av_r = data['av_r'].values()
                self.arefmult = data['arefmult'].values()
                self.art_mult = data['art_mult'].values()
                self.art_match = data['art_match'].values()
                self.aref_beta = data['aref_beta'].values()
                self.ampq = data['ampq'].values()
                self.abetap = data['abetap'].values()
                self.ap_t = data['ap_t'].values()
                self.aphi = data['aphi'].values()
                self.adca = data['adca'].values()
                self.aeta = data['aeta'].values()
                self.arap = data['arap'].values()
                self.anhitsq = data['anhitsq'].values()
                self.anhits_dedx = data['anhits_dedx'].values()
                self.anhitsfit_ratio = data['anhitsfit_ratio'].values()
                self.adedx_pq = data['adedx_pq'].values()
                self.pdedx_pq = data['pdedx_pq'].values()
                self.p_t_p_g = data['p_t_p_g'].values()
                self.msq = data['msq'].values()
                self.hNSigmaProton_0 = data['hNSigmaProton_0'].values()
                self.hNSigmaProton_1 = data['hNSigmaProton_1'].values()
                self.hNSigmaProton_2 = data['hNSigmaProton_2'].values()
                self.hNSigmaProton_3 = data['hNSigmaProton_3'].values()
                self.hNSigmaProton_4 = data['hNSigmaProton_4'].values()
                self.hNSigmaProton_5 = data['hNSigmaProton_5'].values()
                self.hNSigmaProton_6 = data['hNSigmaProton_6'].values()
                self.hNSigmaProton_7 = data['hNSigmaProton_7'].values()
                self.hNSigmaProton_8 = data['hNSigmaProton_8'].values()
                self.hNSigmaProton_9 = data['hNSigmaProton_9'].values()
                self.hNSigmaProton_10 = data['hNSigmaProton_10'].values()
                self.Protons = data['Protons'].values()

        except ValueError:  # Skip empty picos.
            print("ValueError at: " + data_in)  # Identifies the misbehaving file.
        except KeyError:  # Skip non empty picos that have no data.
            print("KeyError at: " + data_in)  # Identifies the misbehaving file.
