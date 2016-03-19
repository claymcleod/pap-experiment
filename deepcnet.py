from __future__ import print_function

import util
import argparse

parser = argparse.ArgumentParser(description='CIFAR10 DeepCNet script for PAP experiment')
parser.add_argument('activation', type=str, help='activation function')
parser.add_argument('nettype', type=str, help='net type - reg, adv, small')
parser.add_argument('-b', '--batchsize', default=128,
                    type=int, help='batch size')
parser.add_argument('-e', '--epochs', default=50,
                    type=int, help='epochs')
parser.add_argument('-l', '--lr', default=0.001,
                    type=float, help='learning_rate')
parser.add_argument('-d', '--dropout', default=None,
                    type=float, help='Dropout rate')
parser.add_argument('-n', '--normalization', default=False,
                    action='store_true', help='batch normalization')
parser.add_argument('-s', '--scheduledlr', default=False,
                    action='store_true', help='Scheduled learning rate (overrides lr)')
args = parser.parse_args()


activation = args.activation
nettype = args.nettype
batch_size = args.batchsize
learning_rate = args.lr
dropout = args.dropout
batch_normalization = args.normalization
epochs = args.epochs
slr = args.scheduledlr
results_file = './cifar10-deepcnet_{}-{}-{}'.format(nettype, activation,
                                                        learning_rate)
initialization = util.get_init_for_activation(activation)
cbs = []
if slr:
    from keras.callbacks import LearningRateScheduler
    def schedule(i):
        if i <= 5:
            return 0.001
        if i <= 20:
            return 0.01
        else:
            return 0.05
    cbs.append(LearningRateScheduler(schedule))
    results_file = results_file+'_scheduled'

cb = util.PersistentHistory(results_file+'.csv')
cbs.append(cb)

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
print("| Initialization: {}".format(initialization))
print("| Scheduled learning rate: {}".format(slr))
print("| File: {}".format(results_file))
print('\\==========================/')
print()


X_train, X_test, Y_train, Y_test = util.get_cifar10()
dcn = util.get_deepcnet(nettype, activation, initialization, dropout, batch_normalization)
util.compile_deepcnet(dcn, learning_rate)


dcn.fit(X_train,
        Y_train,
        batch_size=batch_size,
        nb_epoch=epochs,
        show_accuracy=True,
        shuffle=True,
        validation_data=(X_test, Y_test),
        callbacks=cbs)
