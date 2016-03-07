"""Utilities for PAP thesis."""

import sys
import numpy as np
import theano.tensor as T
import keras
import pandas
import os
from time import time
from keras.layers.core import MaskedLayer, Activation, Dropout, Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.advanced_activations import PReLU
from keras import backend as K
from keras.backend.common import _FLOATX
from keras.datasets import cifar10, cifar100, mnist
from keras.utils import np_utils
from keras.models import Graph, Sequential
from keras.optimizers import SGD


sys.setrecursionlimit(10000)

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def check_session_cores():
    NUM_CORES = os.environ.get('CORES')

    if NUM_CORES:
        sess = tf.Session(
            config=tf.ConfigProto(inter_op_parallelism_threads=int(NUM_CORES),
                                  intra_op_parallelism_threads=int(NUM_CORES)))
        KTF._set_session(sess)
        print("Setting session to have {} cores".format(NUM_CORES))

check_session_cores()

def plot(model, to_file='model.png'):
    from keras.utils.visualize_util import to_graph
    graph = to_graph(model, show_shape=True)
    graph.write_png(to_file)

def get_mnist():
    """Get mnist data."""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    return X_train, X_test, Y_train, Y_test

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
    """Theano step function"""

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
    def __init__(self, activations, bcoefs=None, threshold=False,
                 trainable=True, **kwargs):
        self.activations = activations
        self.bcoefs = bcoefs
        self.threshold = threshold
        self.trainable

        if not self.bcoefs:
            self.bcoefs = [1./len(self.activations)] * len(self.activations)
        assert(len(self.activations) == len(self.bcoefs)),('Coefs != Activations')
        super(ActivationPool, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape[1:]
        self.alphas = []
        for (activation, coef) in zip(self.activations, self.bcoefs):
            init = coef * np.ones(input_shape)
            self.alphas.append(K.variable(init, _FLOATX, None))

        if self.trainable:
            self.trainable_weights = self.alphas

    def get_output(self, train):
        X = self.get_input(train)
        output = 0
        for (activation, bcoef, alpha) in zip(self.activations, self.bcoefs, self.trainable_weights):
            if self.threshold:
                output = output + K.clip(alpha, -bcoef, bcoef) * activation(X)
            else:
                output = output + alpha * activation(X)
        return output

    def get_config(self):
        config = {"name": self.__class__.__name__}
        base_config = super(PReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def mrelu(**kwargs):
    return ActivationPool([T.nnet.relu, step], **kwargs)

def get_activation(model, name):
    if name == 'mrelu':
        model.add(mrelu())
    elif name == 'mrelu-t':
        model.add(mrelu(threshold=True))
    elif name == 'prelu':
        model.add(PReLU())
    elif name == 'relu':
        model.add(Activation('relu'))
    elif name == 'experiment':
        model.add(ActivationPool([K.relu]))
    elif name == 'hrelu':
        model.add(mrelu(trainable=False, bcoefs=[0.5, 0.5]))
    elif name == 'srelu':
        model.add(mrelu(trainable=False, bcoefs=[0.8, 0.2]))
    else:
        print('Invalid activation fn!')
        sys.exit(1)

class PersistentHistory(keras.callbacks.Callback):
    def __init__(self, log_name):
        if os.path.isfile(log_name):
            answer = raw_input("File already exists, would you like to overwrite? (y/N) ")
            if answer.lower() == 'y':
                os.remove(log_name)
            else:
                sys.exit(1)


        self.log_name = log_name

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.accuracies = []
        self.val_accuracies = []
        self.times = []

    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.accs.append(logs.get('acc'))

    def on_epoch_begin(self, batch, logs={}):
        self.loss = []
        self.accs = []
        self.timer = time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time()-self.timer)
        self.losses.append(np.array(self.loss).mean())
        self.val_losses.append(logs.get('val_loss'))
        self.accuracies.append(np.array(self.accs).mean())
        self.val_accuracies.append(logs.get('val_acc'))

        d = {
            'time': self.times,
            'loss': self.losses,
            'acc': self.accuracies,
            'val_loss': self.val_losses,
            'val_acc': self.val_accuracies
        }

        df = pandas.DataFrame.from_dict(d)
        df.to_csv(self.log_name)

def get_mnist_model(activation, lr):
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    get_activation(model, activation)
    model.add(Dropout(0.2))
    model.add(Dense(512))
    get_activation(model, activation)
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model

def get_cifar10_model(activation, lr):
    model = Sequential()
    #model.add(Convolution2D(32, 3, 3, border_mode='same',
    #                    input_shape=(3, 32, 32)))
    #get_activation(model, activation)
    #model.add(Convolution2D(32, 3, 3))
    #get_activation(model, activation)
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    #
    # model.add(Convolution2D(16, 3, 3, border_mode='same'))
    # get_activation(model, activation)
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Flatten(input_shape=(3, 32, 32)))
    for i in range(0, 2):
        model.add(Dense(9216))
        get_activation(model, activation)
        model.add(Dropout(0.05))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model

def build_deepcnet(l, k, activation,
                  first_c3=False,
                  dropout=None,
                  nin=False,
                  final_c1=False):
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(3, 32, 32)))
    if first_c3: model.add(Convolution2D(k, 3, 3, border_mode='same'))
    else: model.add(Convolution2D(k, 2, 2, border_mode='same'))
    get_activation(model, activation)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if nin:
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(k, 1, 1))
        get_activation(model, activation)
    if dropout: model.add(Dropout(dropout))

    for i in range(2, l+1):
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(k*i, 2, 2, border_mode='same'))
        get_activation(model, activation)
        model.add(MaxPooling2D(pool_size=(2, 2)))
        if nin:
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(k*i, 1, 1))
            get_activation(model, activation)
        if dropout: model.add(Dropout(dropout))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(k*(l+1), 2, 2, border_mode='same'))
    get_activation(model, activation)
    if final_c1:
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(k*(l+1), 1, 1))
        get_activation(model, activation)
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model

def compile_deepcnet(model, lr):
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
