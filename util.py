"""Utilities for PAP thesis."""

import os
import sys
import keras
import pandas
import numpy as np
import theano.tensor as T

from time import time
from keras.layers.core import MaskedLayer, Activation, Dropout, Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.backend.common import _FLOATX
from keras.datasets import cifar10, cifar100, mnist
from keras.utils import np_utils
from keras.models import Graph, Sequential
from keras.optimizers import SGD

sys.setrecursionlimit(10000)

def check_session_cores(NUM_CORES):
    import tensorflow as tf
    import keras.backend.tensorflow_backend as KTF
    sess = tf.Session(
        config=tf.ConfigProto(inter_op_parallelism_threads=int(NUM_CORES),
                              intra_op_parallelism_threads=int(NUM_CORES)))
    KTF._set_session(sess)
    print("Setting session to have {} cores".format(NUM_CORES))

NUM_CORES = os.environ.get('CORES')

if NUM_CORES:
    check_session_cores(NUM_CORES)

def plot(model, to_file='model.png'):
    from keras.utils.visualize_util import to_graph
    graph = to_graph(model, show_shape=True)
    graph.write_png(to_file)

def write_dict_as_csv(filename, d):
    if os.path.isfile(filename):
        os.remove(filename)

    df = pandas.DataFrame.from_dict(d)
    df.to_csv(filename, index=False)


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

def relu_integral(x):
    """ReLU piecewise integral"""

    return x**2/2

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

def mrelu(include_d=True, include_i=False, **kwargs):
    act_fns = [T.nnet.relu]
    if include_d: act_fns = act_fns + [step]
    if include_i: act_fns = act_fns + [relu_integral]
    return ActivationPool(act_fns, **kwargs)

def get_activation(model, name, graph=False, i=None, fromnodes=None, blockname=None):
    actfn = None
    if name == 'mrelu':
        actfn = mrelu()
    elif name == 'mrelu-t':
        actfn = mrelu(threshold=True)
    elif name == 'prelu':
        actfn = PReLU()
    elif name == 'relu':
        actfn = Activation('relu')

    if actfn == None:
        print('Invalid activation fn!')
        sys.exit(1)

    if graph:
        node_name = '{}_act{}'.format(blockname, i)
        if isinstance(fromnodes, list):
            model.add_node(actfn, inputs=fromnodes, name=node_name, merge_mode='sum')
        else:
            model.add_node(actfn, input=fromnodes, name=node_name, merge_mode='sum')

        return node_name
    else:
        model.add(actfn)
        return ''

def get_init_for_activation(name):
    if name == 'mrelu' or name == 'mrelu-t':
        return 'uniform'
    else:
        return 'he_uniform'

class PersistentHistory(keras.callbacks.Callback):
    def __init__(self, log_name, check_file=False):
        if os.path.isfile(log_name) and check_file:
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

        write_dict_as_csv(self.log_name, d)

def get_mnist_model(activation, initialization, lr):
    model = Sequential()
    model.add(Dense(512, input_shape=(784,), init=initialization))
    get_activation(model, activation)
    model.add(Dropout(0.2))
    model.add(Dense(512, init=initialization))
    get_activation(model, activation)
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model

def build_deepcnet(l, k, activation, initialization,
                  first_c3=False,
                  dropout=None,
                  nin=False,
                  final_c1=False,
                  batch_normalization=False):
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(3, 32, 32)))
    if first_c3:
        model.add(Convolution2D(k, 3, 3, border_mode='same', init=initialization))
    else:
        model.add(Convolution2D(k, 2, 2, border_mode='same', init=initialization))
    get_activation(model, activation)
    if batch_normalization:
        model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    if nin:
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(k, 1, 1, init=initialization))
        get_activation(model, activation)
        if batch_normalization:
            model.add(BatchNormalization())
    if dropout: model.add(Dropout(dropout))

    for i in range(2, l+1):
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(k*i, 2, 2, border_mode='same', init=initialization))
        get_activation(model, activation)
        if batch_normalization:
            model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        if nin:
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(k*i, 1, 1, init=initialization))
            get_activation(model, activation)
            if batch_normalization:
                model.add(BatchNormalization())
        if dropout: model.add(Dropout(dropout))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(k*(l+1), 2, 2, border_mode='same', init=initialization))
    get_activation(model, activation)
    if batch_normalization:
        model.add(BatchNormalization())
    if final_c1:
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(k*(l+1), 1, 1, init=initialization))
        get_activation(model, activation)
        if batch_normalization:
            model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model

def get_deepcnet(nettype, activation, initialization, dropout, batch_normalization):
    nettype = nettype.lower()
    if nettype == 'reg':
        return build_deepcnet(5, 75, activation, initialization,
                                   dropout=dropout,
                                   final_c1=True,
                                   batch_normalization=batch_normalization)
    elif nettype == 'adv':
        return build_deepcnet(5, 120, activation, initialization,
                                   dropout=dropout,
                                   final_c1=True,
                                   batch_normalization=batch_normalization)
    elif nettype == 'small':
        return build_deepcnet(5, 25, activation, initialization,
                                   dropout=dropout,
                                   final_c1=True,
                                   batch_normalization=batch_normalization)
    else:
        print("Invalid nettype: {}".format(nettype))
        sys.exit(1)

def compile_deepcnet(model, lr):
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)


def build_resnet_block(model, activation, initialization, n, fromnode, blockname,
                       num_filters, kernel_size, change_dim=None):
    lastnode = fromnode
    for i in range(1, n+1):
        model.add_node(Convolution2D(num_filters, kernel_size, kernel_size,
                                     border_mode='same', init=initialization),
                       input=lastnode,
                       name='{}_conv{}'.format(blockname, i))
        model.add_node(BatchNormalization(),
                  input='{}_conv{}'.format(blockname, i),
                  name='{}_bn{}'.format(blockname, i))
        if n == i:
            if change_dim:
                model.add_node(Convolution2D(change_dim, 1, 1,
                                             border_mode='same', init=initialization),
                               input=fromnode,
                               name='{}_change_dim'.format(blockname))
                lastnode = get_activation(model, activation, graph=True, i=i,
                                      fromnodes=['{}_bn{}'.format(blockname, i),
                                                 '{}_change_dim'.format(blockname)],
                                      blockname=blockname)
            else:
                lastnode = get_activation(model, activation, graph=True, i=i,
                                      fromnodes=['{}_bn{}'.format(blockname, i),
                                                 fromnode],
                                      blockname=blockname)
        else:
            lastnode = get_activation(model, activation, graph=True, i=i,
                                      fromnodes='{}_bn{}'.format(blockname, i),
                                      blockname=blockname)

    return lastnode

def build_resnet_34(activation, initialization,  seed=64):
    model = Graph()
    model.add_input(name='input', input_shape=(3, 32, 32))
    model.add_node(ZeroPadding2D((1, 1)), input='input', name='zp')
    model.add_node(Convolution2D(seed, 3, 3, border_mode='same', init='he_normal'),
                    input='zp', name='conv1a')
    model.add_node(MaxPooling2D(pool_size=(1, 1)), input='conv1a', name='mp1')

    conv2a = build_resnet_block(model, activation, initialization, 2, 'mp1', 'conv2a', seed, 3)
    conv2b = build_resnet_block(model, activation, initialization, 2, conv2a, 'conv2b', seed, 3)
    #conv2c = build_resnet_block(model, activation, initialization, 2, conv2b, 'conv2c', seed, 3)
    model.add_node(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)), input=conv2b, name='mp2')

    conv3a = build_resnet_block(model, activation, initialization, 2, 'mp2', 'conv3a', seed*2, 3, change_dim=seed*2)
    conv3b = build_resnet_block(model, activation, initialization, 2, conv3a, 'conv3b', seed*2, 3)
    #conv3c = build_resnet_block(model, activation, initialization, 2, conv3b, 'conv3c', seed*2, 3)
    #conv3d = build_resnet_block(model, activation, initialization, 2, conv3c, 'conv3d', seed*2, 3)
    model.add_node(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)), input=conv3b, name='mp3')

    conv4a = build_resnet_block(model, activation, initialization, 2, 'mp3', 'conv4a', seed*4, 3, change_dim=seed*4)
    conv4b = build_resnet_block(model, activation, initialization, 2, conv4a, 'conv4b', seed*4, 3)
    #conv4c = build_resnet_block(model, activation, initialization, 2, conv4b, 'conv4c', seed*4, 3)
    #conv4d = build_resnet_block(model, activation, initialization, 2, conv4c, 'conv4d', seed*4, 3)
    #conv4e = build_resnet_block(model, activation, initialization, 2, conv4d, 'conv4e', seed*4, 3)
    model.add_node(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)), input=conv4b, name='mp4')

    conv5a = build_resnet_block(model, activation, initialization, 2, 'mp4', 'conv5a', seed*8, 3, change_dim=seed*8)
    conv5b = build_resnet_block(model, activation, initialization, 2, conv5a, 'conv5b', seed*8, 3)
    #conv5c = build_resnet_block(model, activation, initialization, 2, conv5b, 'conv5c', seed*8, 3)

    model.add_node(Flatten(), input=conv5b, name='flatten')
    model.add_node(Dense(1000, init=initialization), input='flatten', name='fc1000')
    fc1000act = get_activation(model, activation, graph=True, i='', fromnodes='fc1000', blockname='dc1000_act')
    model.add_node(Dense(100, init=initialization), input=fc1000act, name='fc100')
    model.add_output(name='output', input='fc100')
    return model

def compile_resnet(model, lr):
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss={'output':'categorical_crossentropy'}, optimizer=sgd)
