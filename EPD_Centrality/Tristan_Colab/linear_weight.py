#
# \brief Generates weights using the linear weighting method
#
# \author Tristan Protzman
# \date 11/01/2021
# \email tlprotzman@gmail.com
# \affiliation Lehigh University
#
# /

from typing import \
    TYPE_CHECKING  # hack to avoid circular dependency, see https://www.stefaanlippens.net/circular-imports-type-hints-python.html

if TYPE_CHECKING:
    from training import CentralityModel

import logging
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def generate_weights(data: 'CentralityModel') -> np.array:
    A = np.zeros((17, 17))
    B = np.zeros(17)
    if not os.path.exists("cache"):
        os.makedirs("cache")
    if os.path.isfile("cache/weights.npy"):  # Some dead simple caching to speed up development
        logging.warning(
            "Loading cached weights")  # TODO Make this caching a bit more robust, handling different file sets and such, really should be done before even loading the ROOT file
        return np.load("cache/weights.npy")

    A[:16, :16] = np.matmul(data.training_input.T, data.training_input)
    A[16][:16] = np.sum(data.training_input, axis=0)
    A[16][16] = data.num_events
    logging.info("Generated A")

    B[:16] = np.matmul(data.training_target.T, data.training_input)
    B[16] = np.sum(data.training_target)
    logging.info("Generated B")

    logging.debug("A Matrix:")
    logging.debug(A)
    logging.debug("B Matrix:")
    logging.debug(B)

    weights = np.linalg.solve(A, B)
    logging.info("Generated Linear Weights")
    logging.debug("Weights:")
    logging.debug(weights)
    np.save("cache/weights.npy", weights)
    return weights

# TODO Make the savefile directory an input for main.
# TODO Make it trivial to apply this to a different input set, rather than what it was trained on
def apply_weights(data: 'CentralityModel', application_data: dict) -> np.array:
    logging.info("Appling linear weights")
    predict = None
    if application_data is None:
        predict = np.zeros(data.num_events)
        for i in range(len(data.training_input)):
            predict[i] = np.dot(data.model_info["linear_weight"][:-1], data.training_input[i]) + \
                         data.model_info["linear_weight"][-1]
    else:
        predit = np.zeros(len(application_data["data"]))
        for i in range(len(application_data["data"])):
            predict[i] = np.dot(data.model_info["linear_weight"][:-1], application_data["data"][i]) + \
                         data.model_info["linear_weight"][-1]
    logging.debug("Predicted Values:")
    logging.debug(predict)

    # plt.hist(predict)
    # plt.tight_layout()
    # plt.show()
    # plt.hist(data.training_evaluation)
    # plt.tight_layout()
    # plt.show()
    truth = data.training_evaluation
    if application_data is not None:
        truth = application_data["truth"]

    level = logging.getLogger().getEffectiveLevel()  # Don't care about all the info/debug stuff in matplotlib
    logging.getLogger().setLevel(logging.WARNING)
    plt.hist2d(truth, predict, bins=(300, 320), range=[[0, 300], [0, 16]], norm=colors.LogNorm())
    plt.ylabel("Predicted impact parameter")
    plt.xlabel("TPC RefMult")
    plt.title("Linear Weighted Impact Parameter Predictions")
    plt.colorbar()
    plt.savefig(r"C:\Users\dansk\Documents\Thesis\Tristan\hist.png")
    logging.getLogger().setLevel(level)

    return predict
