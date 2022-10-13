# Functions for ml_generator.py

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, Conv1D
from keras.utils.generic_utils import get_custom_objects
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import mixture
import matplotlib.pyplot as plt


"""
This forst section is for a supervised, MLP NN. This is used when
aiming your "fit" at a target; for instance, at weighting EPD range
nMIP (or particle count) at TPC range reference multiplicity.

A linear weighting can be had by simply using the linear method,
which is weighting (with a bias term) without a specific neuron
type and using only a single neuron.
"""


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


class Bose(Activation):

    def __init__(self, activation, **kwargs):
        super(Bose, self).__init__(activation, **kwargs)
        self.__name__ = 'Bose'


def bose(x):
    return x/K.exp(x)


get_custom_objects().update({'Bose': Bose(bose)})


def lr_scheduler(epoch, lr):
    """
    This function varies the learning rate throughout the
    training so that it will not get stuck on a saddle point.
    """
    decay_rate = 0.995
    decay_step = 3
    if epoch % decay_step == 0 and epoch:
        return lr * pow(decay_rate, np.floor(epoch / decay_step))
    return lr


def ml_run(rings, target, tar_str, actFunc='Swish', loss='logcosh', optimizer='Nadam',
           metrics='mae', monitor='val_loss', min_delta=0.01, patience=8,
           validation_split=0.2, epochs=100, batch_size=3000, h_lay=2,
           layer1=32, layer2=128, layer3=256, actFunc2='relu', mult_func=False,
           conv_layer1=128, conv_layer2=256, CNN=False):
    input_dim = len(rings[0])

    if tar_str == 'b':
        acc_goal = 0.55
    else:
        acc_goal = 32.0

    if actFunc == 'CNN':
        CNN = True
        batch_size = batch_size * 3
        patience = 3
        actFunc = 'relu'

    print('########################################',
          '\n \n \n',
          'Activation Function:', actFunc,
          '\n \n \n',
          'Target:', tar_str,
          '\n \n \n',
          '########################################')

    model = Sequential()

    """
    This is to use the variable learning rate; I need to
    make this work with different types of optimizers.
    """
    opt = keras.optimizers.Nadam(learning_rate=0.001)
    if mult_func is False:
        actFunc2 = actFunc
    if CNN is False:
        if actFunc == 'linear':
            model.add(Dense(1, input_dim=input_dim))
        else:
            model.add(Dense(layer1, input_shape=(input_dim,), activation='{0}'.format(actFunc)))
            model.add(Dense(layer2, activation='{0}'.format(actFunc)))
            if h_lay > 2:
                for i in range(h_lay-2):
                    model.add(Dense(layer3, activation='{0}'.format(actFunc2)))
            model.add(Dense(1, activation='{0}'.format(actFunc)))
    else:
        rings = np.expand_dims(rings, axis=2)
        model = Sequential()
        model.add(Conv1D(conv_layer1, kernel_size=3, activation='{0}'.format(
            actFunc), input_shape=(input_dim, 1)))
        model.add(Conv1D(conv_layer2, 5, activation='{0}'.format(actFunc)))
        model.add(Flatten())
        model.add(Dense(512, activation='{0}'.format(actFunc)))
        model.add(Dense(1, activation='{0}'.format(actFunc)))
    model.compile(loss=loss,
                  optimizer=opt, metrics=[metrics])
    early_stopping = EarlyStopping(
        monitor=monitor, min_delta=min_delta, patience=patience,
        restore_best_weights=True)
    model.fit(rings, target, validation_split=validation_split,
              callbacks=[early_stopping, LearningRateScheduler(lr_scheduler, verbose=1)],
              epochs=epochs, batch_size=batch_size)
    _, accuracy = model.evaluate(rings, target)
    if accuracy < acc_goal:
        print("Accuracy: {}".format(accuracy), '. Winner winner chicken dinner!')
    else:
        print("Accuracy: {}".format(accuracy), '. I\'m not mad; I\'m just disappointed.')

    weights = []
    biases = []
    for i in range(len(model.layers)):
        weights.append(model.layers[i].get_weights()[0])
        biases.append(model.layers[i].get_weights()[1])
    weights = np.asarray(weights)
    biases = np.asarray(biases)

    predictions = model.predict(rings).flatten()
    # np.save(file_loc + str(energy) + "{}_predictions.npy".format(actFunc), predictions)

    return predictions, model, weights, biases


"""
The following are unsupervised algorithms. The advantage for these is that
we can train centrality using a self-referential metric, like just nMIP from
the EPD range or just RefMult from the TPC range. The disadvantage in
practise, of course, is the lack of any unbiased vetting of the results.

All of the following are cluster-based algorithms.
"""


# This is a KMeans model, with clusters n.
def model_kmeans(data, n_clusters=5, n_init=10, max_iter=300):
    model = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)
    model.fit(data)
    predictions = model.predict(data)
    return predictions


# KMeans Minibach model, with clusters n.
def model_kmeans_minbatch(data, n_clusters=10, n_init=10, max_iter=300):
    model = MiniBatchKMeans(init='random', n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)
    model.fit(data)
    predictions = model.predict(data)
    return predictions


# Gausian Mixture Model (GMM; form of k-means).
def model_gmm(data, n_clusters=10, n_init=10, cov_type='diag', max_iter=10):
    model = mixture.GaussianMixture(n_components=n_clusters, covariance_type=cov_type,
                                    n_init=n_init, max_iter=max_iter)
    model.fit(data)
    predictions = model.predict(data)
    return predictions