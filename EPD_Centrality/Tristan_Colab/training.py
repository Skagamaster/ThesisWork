import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot as up
from os import error
import typing
import logging
import linear_weight
import glauber


class CentralityModel:

    def __init__(self, simulated_data: bool) -> None:
        self.simulation: bool
        self.simulation = simulated_data
        self.model_types = {
            "linear_weight": {"train": linear_weight.generate_weights, "apply": linear_weight.apply_weights},
            "glauber_model": {"train": glauber.generate_weights, "apply": glauber.apply_weights}}
        self.model_info = dict()
        self.model_predictions = dict()
        self.training_input = None
        self.num_events = None
        self.training_target = None
        self.training_evaluation = None

    def import_data(self, data_input):
        """Loads the data from the specified root file into memory,
           allowing the model to be created
        Args:
            data_input (str): The path to the root file containing the summary
                             of the ring values and the impact parameter/refmult
        """
        # Path specified in main.py. If standalone testing, uncomment the following with your input path:
        # data_input = r"C:\Users\dansk\Downloads\simulated_data.root"
        in_file = up.open(data_input)

        # Read in the inputs to the algorithm, the sums in each ring in the EPD
        data_len = len(in_file['ring_sums'].member("fElements"))
        if data_len % 16 is False:
            raise ValueError("16 rows were not found importing training data")
        data_len = int(data_len/16)
        self.training_input = np.reshape(in_file["ring_sums"].member("fElements"), (16, data_len)).T
        logging.info("EPD data imported")

        self.num_events = len(self.training_input)

        # Read in the target we would like to fit to, the impact parameter
        if self.simulation:
            self.training_target = in_file["impact_parameter"].member("fElements")
        else:
            self.training_target = in_file["tpc_multiplicity"].member("fElements")
        if len(self.training_target) != self.num_events:
            raise ValueError("Number of events are different for ring data and b/refmult")

        logging.info("Target data read in")

        # Read in the parameter we would like to evaluate against, currently refmult from the TPC
        if not self.simulation:
            self.training_evaluation = self.training_target  # If we are using this with detector data, the impact parameter is not known
        else:
            self.training_evaluation = in_file["tpc_multiplicity"].member("fElements")
            if len(self.training_evaluation) != self.num_events:
                raise ValueError("Number of events are different for ring data and refmult")

        logging.info("Basic evaluation data read in")

        logging.debug("Training Data:")
        logging.debug(self.training_input)
        logging.debug("Target Data:")
        logging.debug(self.training_target)
        logging.debug("Evaluation Data")
        logging.debug(self.training_evaluation)

    def build_model(self, m_type: str) -> None:
        """build_model builds a set of weights which can be applied to input data to determine the centrality of an event.  Currently
           only a linear weight model is implemented, but hopefully more will come.
        Args:
            m_type (str): Which model should be used to generate the weights.  Current choices are "linear_weight", "glauber_model"
        Raises:
            KeyError: Raises a KeyError if the type is not an implemented method.
        """

        if m_type not in self.model_types.keys():
            raise KeyError('No model "' + m_type + '" known')
        logging.info("Building model: " + m_type)
        self.model_info[m_type] = self.model_types[m_type]["train"](
            self)  # generates the desired model and saves the info needed to apply it

    def apply_model(self, m_type: str, application_data: dict = None) -> None:
        """apply_model applys the model created by build_model to either the already loaded evaluation data, or to another
           data set if it is specified
        Args:
            m_type (str): The model to be be applied.  Current choices are "linear_weight", "glauber_model"
            application_data (dict, optional): The data set to be applied.  Expects the key "data" to be a numpy array with the ring data,
                                               and the key "truth" with the expected values. Defaults to None.
        Raises:
            KeyError: Raises a KeyError if the type is not an implemented method.
        """
        if m_type not in self.model_types.keys():
            raise KeyError('No model "' + m_type + '" known')

        if application_data is not None:
            if "data" not in application_data.keys() or "truth" not in application_data.keys():
                raise KeyError("Missing data or truth values for application data")

        logging.info("Applying model " + m_type)
        self.model_predictions[m_type] = self.model_types[m_type]["apply"](self, application_data)  # Apply the model

    def evaluate_model(self, m_type: str) -> None:
        quantiles = np.arange(0, 1.01, 0.05)
        logging.debug("Finding quantiles:")
        logging.debug(quantiles)

        predict_quantiles = np.quantile(self.model_predictions[m_type], quantiles)
        logging.debug("Predicted values quantiles are:")
        logging.debug(predict_quantiles)

        eval_quantiles = np.quantile(self.training_evaluation, quantiles)
        logging.debug("Evaluation values quantiles are:")
        logging.debug(eval_quantiles)

        # def filter()
