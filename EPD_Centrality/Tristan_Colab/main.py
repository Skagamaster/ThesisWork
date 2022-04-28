# \brief Handles running the analysis of the supplied input data as
#        well as applying it to actual data sets
#
# \author Tristan Protzman
# \date 01/10/2021
# \email tlprotzman@gmail.com
# \affiliation Lehigh University
#
#

import sys
import typing
import logging
import time


import training

time_start = time.perf_counter()


def main(args):
    # Training Phase
    model_type = "linear_weight"  # Which model to use
    model = training.CentralityModel(True)
    logging.info("Created model object")

    model.import_data(r"C:\Users\dansk\Downloads\simulated_data.root")
    logging.info("Imported data from ROOT file")

    model.build_model(model_type)
    logging.info("Generated model")

    # Application Phase
    model.apply_model(model_type)
    logging.info("Applied Model")
    model.evaluate_model(model_type)
    print(time.perf_counter() - time_start)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s [%(filename)s:%(lineno)d] %(message)s')
    main(sys.argv)

