# Functions for ml_generator.py

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, Conv1D
from keras.utils.generic_utils import get_custom_objects
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras.callbacks import EarlyStopping
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import mixture


class Swish(Activation):

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'


def swish(x, beta=2):
    return x * K.sigmoid(beta * x)


get_custom_objects().update({'swish': Swish(swish)})


class Mish(Activation):

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'mish'


def mish(x):
    return x * K.tanh(K.softplus(x))


get_custom_objects().update({'mish': Mish(mish)})


class Bose(Activation):

    def __init__(self, activation, **kwargs):
        super(Bose, self).__init__(activation, **kwargs)
        self.__name__ = 'bose'


def bose(x):
    return x/K.exp(x)


get_custom_objects().update({'bose': Bose(bose)})


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


def ml_run(rings, refmult3, actFunc='swish', loss='logcosh', optimizer='Nadam',
           metrics='mae', monitor='val_loss', min_delta=0.01, patience=8,
           validation_split=0.2, epochs=100, batch_size=3000, energy=19, CNN=False,
           file_loc=r'C:\Users\dansk\Documents\Thesis\2021_DNP_Work\ML_fits\9_2021\\'):
    input_dim = len(rings[0])
    if actFunc == 'mish':
        CNN = True
        batch_size = 150
    model = Sequential()
    if CNN is False:
        if actFunc == 'linear':
            model.add(Dense(1, input_dim=input_dim))
        else:
            model.add(Dense(32, input_shape=(input_dim,), activation='{0}'.format(actFunc)))
            model.add(Dense(128, activation='{0}'.format(actFunc)))
            model.add(Dense(1, activation='{0}'.format(actFunc)))
    else:
        rings = np.expand_dims(rings, axis=2)
        model = Sequential()
        model.add(Conv1D(128, kernel_size=3, activation='{0}'.format(
            actFunc), input_shape=(input_dim, 1)))
        model.add(Conv1D(256, 5, activation='{0}'.format(actFunc)))
        model.add(Flatten())
        model.add(Dense(512, activation='{0}'.format(actFunc)))
        model.add(Dense(1, activation='{0}'.format(actFunc)))
    model.compile(loss=loss,
                  optimizer=optimizer, metrics=[metrics])
    early_stopping = EarlyStopping(
        monitor=monitor, min_delta=min_delta, patience=patience,
        restore_best_weights=True)
    model.fit(rings, refmult3, validation_split=validation_split, callbacks=[early_stopping],
              epochs=epochs, batch_size=batch_size)
    _, accuracy = model.evaluate(rings, refmult3)
    if accuracy < 25:
        print("Accuracy: {}".format(accuracy), '. You win!')
    else:
        print("Accuracy: {}".format(accuracy), '. Your life is a lie.')
    """
    weights = []
    biases = []
    for i in range(len(model.layers)):
        weights.append(model.layers[i].get_weights()[0])
        biases.append(model.layers[i].get_weights()[1])
    weights = np.asarray(weights)
    biases = np.asarray(biases)

    model.save(file_loc + str(energy) + "{0}Fitmodel.h5".format(actFunc))

    for i in range(len(weights)):
        np.savetxt(file_loc + str(energy) + "{0}Weights{1}.txt".format(actFunc, i),
                   weights[i], delimiter=',', newline='}, \n {',
                   fmt='%f')
    for i in range(len(biases)):
        np.savetxt(file_loc + str(energy) + "{0}Biases{1}.txt".format(actFunc, i),
                   biases[i], delimiter=',', newline=",",
                   fmt='%f')

    plot_model(model, to_file=file_loc + str(energy) + "{0}Model.png".format(actFunc), show_shapes=True)
    """
    predictions = model.predict(rings).flatten()
    # np.save(file_loc + str(energy) + "{}_predictions.npy".format(actFunc), predictions)

    return predictions
