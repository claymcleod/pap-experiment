from __future__ import print_function
import argparse
from keras.callbacks import LearningRateScheduler

parser = argparse.ArgumentParser(description='CIFAR10 DeepCNet script for PAP experiment')
parser.add_argument('activation', type=str, help='activation function')
parser.add_argument('-b', '--batchsize', default=128,
                    type=int, help='batch size')
parser.add_argument('-e', '--epochs', default=500,
                    type=int, help='epochs')
args = parser.parse_args()

learning_rate = 0.001
activation = args.activation
batch_size = args.batchsize
epochs = args.epochs

import util
print()
print('/==========================\\')
print("| Dataset: CIFAR10 (DeepCNet_small)")
print("| Activation: {}".format(activation))
print("| Batch size: {}".format(batch_size))
print("| Epochs: {}".format(epochs))
print('\\==========================/')
print()

X_train, X_test, Y_train, Y_test = util.get_cifar10()
dcn = util.build_deepcnet(5, 25, activation, final_c1=True)
util.compile_deepcnet(dcn, learning_rate)

cb = util.PersistentHistory('./cifar10-deepcnet_small-{}-{}.csv'.format(activation, learning_rate))
lrcb = LearningRateScheduler(lambda x: 0.001 + 0.001*x)

dcn.fit(X_train, Y_train,
        batch_size=batch_size, nb_epoch=epochs,
        show_accuracy=True,
        shuffle=True,
        validation_data=(X_test, Y_test),
        callbacks=[cb, lrcb])
