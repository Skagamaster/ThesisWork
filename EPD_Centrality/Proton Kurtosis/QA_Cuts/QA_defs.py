import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema as arex
from scipy.signal import savgol_filter as sgf
import uproot as up
import awkward as ak
import time


class PicoDST:
    """This class makes the PicoDST from the root file, along with
    all of the observables I use for proton kurtosis analysis."""

    def __init__(self, data: bool) -> None:
        """This defines the variables we'll be using
        in the class."""
        self.data: bool
        self.adc = None
        self.smooth_adc = None
        self.adc_max = None
        self.adc_max_mean = None
        self.adc_argmax = None
        self.adc_extrema = None
        self.tile_names = None
        self.ped_tiles = None
        self.ped_tiles_loc = None

    def import_data(self, data_in):
        """This imports the data. You must have the latest versions
            of uproot and awkward installed on your machine (uproot4 and
            awkward 1.0 as of the time of this writing).
            Use: pip install uproot awkward
        Args:
            data_in (str): The path to the picoDst ROOT file"""
        try:
            data = up.open(data_in)["PicoDst"]
        except ValueError:  # Skip empty picos.
            print("ValueError at", data_in)  # Identifies the misbehaving file.
        except KeyError:  # Skip non empty picos that have no data.
            print("KeyError at", data_in)  # Identifies the misbehaving file.
