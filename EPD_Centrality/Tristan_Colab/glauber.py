#
# \brief
#
# \author Tristan Protzman
# \date
# \email tlprotzman@gmail.com
# \affiliation Lehigh University
#
#


from typing import Any, \
    TYPE_CHECKING  # hack to avoid circular dependency, see https://www.stefaanlippens.net/circular-imports-type-hints-python.html

if TYPE_CHECKING:
    from training import CentralityModel

import numpy as np

import logging


def generate_weights(data: 'centrality_model') -> any:
    pass


def apply_weights(data: 'centrality_model', application_data: any) -> np.array:
    pass
