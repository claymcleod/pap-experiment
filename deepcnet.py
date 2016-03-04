from __future__ import print_function
import argparse

parser = argparse.ArgumentParser(description='CIFAR10 DeepCNet script for PAP experiment')
parser.add_argument('activation', type=str, help='activation function')
parser.add_argument('-l', '--learningrate',default=0.01,
                    type=float, help='learning rate')
parser.add_argument('-b', '--batchsize', default=32,
                    type=int, help='batch size')
parser.add_argument('-e', '--epochs', default=100,
                    type=int, help='epochs')
parser.add_argument('-d','--augmentation', default=False, action='store_true')
args = parser.parse_args()

data_augmentation = args.augmentation
activation = args.activation
learning_rate = args.learningrate
batch_size = args.batchsize
epochs = args.epochs

import util
print()
print('/==========================\\')
print("| Dataset: CIFAR10 (DeepCNet)")
print("| Activation: {}".format(activation))
print("| Learning rate: {}".format(learning_rate))
print("| Batch size: {}".format(batch_size))
print("| Epochs: {}".format(epochs))
print('| Data augmentation: {}'.format(data_augmentation))
print('\\==========================/')
print()

X_train, X_test, Y_train, Y_test = util.get_cifar10()
dcn = util.build_deepcnet(4, 100, dropout=0.2, nin=True)
util.compile_deepcnet(dcn, 'relu', 0.01)

dcn.fit(X_train, Y_train,
          batch_size=batch_size, nb_epoch=epochs,
          show_accuracy=True,
          shuffle=True,
          validation_data=(X_test, Y_test),
          callbacks=[util.PersistentHistory('./cifar10-deepcnet-{}-{}.csv'.format(activation, learning_rate))])
