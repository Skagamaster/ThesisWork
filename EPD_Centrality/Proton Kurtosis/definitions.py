import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils.generic_utils import get_custom_objects
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras.callbacks import EarlyStopping

# Gets ew, pp, and tt from EPD ID


def EPDTile(tile=int(123)):
    if tile < 0:
        ew = 0
    else:
        ew = 1
    pp = int(abs(tile/100))
    tt = abs(tile) % 100
    return [ew, pp, tt]

# Gets the ring position from the EPD ID


def EPDRing(tile=int(123)):
    if tile < 0:
        ew = 0
    else:
        ew = 1
    row = int((abs(tile) % 100)/2)+1+ew*16
    return [ew, row]

# Here's a Swish activation function in case you want to use it.


class Swish(Activation):

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'


def swish(x, beta=2):
    return (x * K.sigmoid(beta * x))


get_custom_objects().update({'swish': Swish(swish)})


class Mish(Activation):

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'mish'


def mish(x):
    return (x * K.tanh(K.softplus(x)))


get_custom_objects().update({'mish': Mish(mish)})


class Bose(Activation):

    def __init__(self, activation, **kwargs):
        super(Bose, self).__init__(activation, **kwargs)
        self.__name__ = 'bose'


def bose(x):
    return x/K.exp(x)


get_custom_objects().update({'bose': Bose(bose)})
