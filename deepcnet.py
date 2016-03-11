from __future__ import print_function

import util
import argparse

parser = argparse.ArgumentParser(description='CIFAR10 DeepCNet script for PAP experiment')
parser.add_argument('activation', type=str, help='activation function')
parser.add_argument('nettype', type=str, help='net type - reg, adv, small')
parser.add_argument('-b', '--batchsize', default=128,
                    type=int, help='batch size')
parser.add_argument('-e', '--epochs', default=100,
                    type=int, help='epochs')
parser.add_argument('-l', '--lr', default=0.002,
                    type=int, help='learning_rate')
parser.add_argument('-d', '--dropout', default=None,
                    type=float, help='Dropout rate')
parser.add_argument('-n', '--normalization', default=False,
                    action='store_true', help='batch normalization')
args = parser.parse_args()


activation = args.activation
nettype = args.nettype
batch_size = args.batchsize
learning_rate = args.lr
dropout = args.dropout
batch_normalization = args.normalization
epochs = args.epochs
results_file = './cifar10-deepcnet_{}-{}-{}.csv'.format(nettype, activation,
                                                        learning_rate)
print()
print('/==========================\\')
print("| Dataset: CIFAR10 (DeepCNet)")
print("| Net type: {}".format(nettype))
print("| Activation: {}".format(activation))
print("| Learning rate: {}".format(learning_rate))
print("| Batch size: {}".format(batch_size))
print("| Batch normalization: {}".format(batch_normalization))
print("| Dropout: {}".format(dropout))
print("| Epochs: {}".format(epochs))
print('\\==========================/')
print()


X_train, X_test, Y_train, Y_test = util.get_cifar10()
dcn = util.get_deepcnet(nettype, activation, dropout, batch_normalization)
util.compile_deepcnet(dcn, learning_rate)
cb = util.PersistentHistory(results_file)
dcn.fit(X_train,
        Y_train,
        batch_size=batch_size,
        nb_epoch=epochs,
        show_accuracy=True,
        shuffle=True,
        validation_data=(X_test, Y_test),
        callbacks=[cb])
