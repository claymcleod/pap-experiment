"""Utilities for PAP thesis."""

import numpy as np
from keras.layers.core import MaskedLayer, Activation, Dropout, Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import backend as K
from keras.backend.common import _FLOATX
from keras.datasets import cifar10, cifar100
from keras.utils import np_utils
from keras.models import Graph


def get_cifar10():
    """Get cifar10 data."""
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    return X_train, X_test, Y_train, Y_test


def get_cifar100():
    """Get cifar100 data."""
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    Y_train = np_utils.to_categorical(y_train, 100)
    Y_test = np_utils.to_categorical(y_test, 100)
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    return X_train, X_test, Y_train, Y_test

def step(x):
    return K.switch(x > 0, 1, 0)

class Step(MaskedLayer):
    """Step activation module."""

    def __init__(self, **kwargs):
        super(Step, self).__init__(**kwargs)
        self.activation = step

    def get_output(self, train=False):
        X = self.get_input(train)
        return self.activation(X)

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'activation': self.activation.__name__}
        base_config = super(Activation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ActivationPool(MaskedLayer):
    def __init__(self, activations, **kwargs):
        self.activations = activations
        super(ActivationPool, self).__init__(**kwargs)

    def build(self):
        #input_shape = self.input_shape[1:]
        coef = 1./len(self.activations)
        init = coef * np.ones(shape)
        print(init)
        #self.alphas = K.variable(np.ones(shape), dtype, name)
        #self.trainable_weights = [self.alphas]

        #if self.initial_weights is not None:
        #    self.set_weights(self.initial_weights)
        #    del self.initial_weights

    #def get_output(self, train):
    #    X = self.get_input(train)
    #    pos = K.relu(X)
    #    neg = self.alphas * (X - abs(X)) * 0.5
    #    return pos + neg

    #def get_config(self):
    #    config = {"name": self.__class__.__name__,
    #              "init": self.init.__name__}
    #    base_config = super(PReLU, self).get_config()
    #    return dict(list(base_config.items()) + list(config.items()))

def get_nonpap_model(channels, rows, cols, classes, loss, optimizer):
    model = Graph()
    model.add_input(name='input', input_shape=(channels, rows, cols))
    model.add_node(Convolution2D(32, 3, 3, border_mode='same'), input='input',
                   name='conv1_a')
    model.add_node(Activation('relu'), input='conv1_a', name='relu1_a')
    model.add_node(Convolution2D(32, 3, 3), input='relu1_a',
                   name='conv1_b')
    model.add_node(Activation('relu'), input='conv1_b', name='relu1_b')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   input='relu1_b', name='mp1')
    model.add_node(Dropout(0.25), input='mp1', name='do1')

    model.add_node(Convolution2D(64, 3, 3, border_mode='same'), input='do1',
                   name='conv2_a')
    model.add_node(Activation('relu'), input='conv2_a', name='relu2_a')
    model.add_node(Convolution2D(64, 3, 3), input='relu2_a',
                   name='conv2_b')
    model.add_node(Activation('relu'), input='conv2_b', name='relu2_b')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   input='relu2_b', name='mp2')
    model.add_node(Dropout(0.25), input='mp2', name='do2')

    model.add_node(Flatten(), input='do2', name='flatten')
    model.add_node(Dense(512), input='flatten', name='d1')
    model.add_node(Activation('relu'), input='d1', name='relu_out')
    model.add_node(Dropout(0.5), input='relu_out', name='d2')
    model.add_node(Dense(classes), input='d2', name='d')
    model.add_node(Activation('softmax'), input='d', name='sm')
    model.add_output(name='output', input='sm')
    model.compile(loss={'output': loss}, optimizer=optimizer)
    return model

def get_pap_model(channels, rows, cols, classes, loss, optimizer):
    model = Graph()
    model.add_input(name='input', input_shape=(channels, rows, cols))
    model.add_node(Convolution2D(32, 3, 3, border_mode='same'), input='input',
                   name='conv1_a')
    model.add_node(Activation('relu'), input='conv1_a', name='relu1_a')
    model.add_node(Step(), input='conv1_a', name='step1_a')
    model.add_node(Convolution2D(32, 3, 3), inputs=['relu1_a', 'step1_a'],
                   name='conv1_b')
    model.add_node(Activation('relu'), input='conv1_b', name='relu1_b')
    model.add_node(Step(), input='conv1_b', name='step1_b')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   inputs=['relu1_b', 'step1_b'], name='mp1')
    model.add_node(Dropout(0.25), input='mp1', name='do1')

    model.add_node(Convolution2D(64, 3, 3, border_mode='same'), input='do1',
                   name='conv2_a')
    model.add_node(Activation('relu'), input='conv2_a', name='relu2_a')
    model.add_node(Step(), input='conv2_a', name='step2_a')
    model.add_node(Convolution2D(64, 3, 3), inputs=['relu2_a', 'step2_a'],
                   name='conv2_b')
    model.add_node(Activation('relu'), input='conv2_b', name='relu2_b')
    model.add_node(Step(), input='conv2_b', name='step2_b')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   inputs=['relu2_b', 'step2_b'], name='mp2')
    model.add_node(Dropout(0.25), input='mp2', name='do2')

    model.add_node(Flatten(), input='do2', name='flatten')
    model.add_node(Dense(512), input='flatten', name='d1')
    model.add_node(Activation('relu'), input='d1', name='relu_out')
    model.add_node(Dropout(0.5), input='relu_out', name='d2')
    model.add_node(Dense(classes), input='d2', name='d')
    model.add_node(Activation('softmax'), input='d', name='sm')
    model.add_output(name='output', input='sm')
    model.compile(loss={'output': loss}, optimizer=optimizer)
    return model


def get_semipap_model(channels, rows, cols, classes, loss, optimizer):
    model = Graph()
    model.add_input(name='input', input_shape=(channels, rows, cols))
    model.add_node(Convolution2D(32, 3, 3, border_mode='same'), input='input',
                   name='conv1_a')
    model.add_node(Activation('relu'), input='conv1_a', name='relu1_a')
    model.add_node(Step(), input='conv1_a', name='step1_a')
    model.add_node(Convolution2D(32, 3, 3), input='relu1_a',
                   name='conv1_b')
    model.add_node(Activation('relu'), input='conv1_b', name='relu1_b')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   input='relu1_b', name='mp1')
    model.add_node(Dropout(0.25), input='mp1', name='do1')

    model.add_node(Convolution2D(64, 3, 3, border_mode='same'), input='do1',
                   name='conv2_a')
    model.add_node(Activation('relu'), input='conv2_a', name='relu2_a')
    model.add_node(Convolution2D(64, 3, 3), input='relu2_a',
                   name='conv2_b')
    model.add_node(Activation('relu'), input='conv2_b', name='relu2_b')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   input='relu2_b', name='mp2')
    model.add_node(Dropout(0.25), input='mp2', name='do2')

    model.add_node(Flatten(), input='do2', name='flatten')
    model.add_node(Dense(512), input='flatten', name='d1')
    model.add_node(Activation('relu'), input='d1', name='relu_out')
    model.add_node(Dropout(0.5), input='relu_out', name='d2')
    model.add_node(Dense(classes), input='d2', name='d')
    model.add_node(Activation('softmax'), input='d', name='sm')
    model.add_output(name='output', input='sm')
    model.compile(loss={'output': loss}, optimizer=optimizer)
    return model
