from __future__ import print_function

import util
import argparse
from keras.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser(description='CIFAR100 Resnet script for PAP experiment')
parser.add_argument('activation', type=str, help='activation function')
parser.add_argument('-b', '--batchsize', default=256,
                    type=int, help='batch size')
parser.add_argument('-e', '--epochs', default=100,
                    type=int, help='epochs')
parser.add_argument('-l', '--lr', default=0.001,
                    type=float, help='learning_rate')
args = parser.parse_args()


activation = args.activation
batch_size = args.batchsize
learning_rate = args.lr
epochs = args.epochs
results_file = './cifar100-resnet_{}-{}.csv'.format(activation, learning_rate)
initialization = util.get_init_for_activation(activation)
print()
print('/==========================\\')
print("| Dataset: CIFAR100 (Resnet)")
print("| Activation: {}".format(activation))
print("| Learning rate: {}".format(learning_rate))
print("| Batch size: {}".format(batch_size))
print("| Epochs: {}".format(epochs))
print("| Initialization: {}".format(initialization))
print('\\==========================/')
print()


X_train, X_test, Y_train, Y_test = util.get_cifar100()
print("Building...")
resnet = util.build_resnet_34(activation, initialization, seed=32)
print("Compiling...")
util.compile_resnet(resnet, learning_rate)
cb = util.PersistentHistory(results_file)
cb2 = ModelCheckpoint('./cifar100-resnet_{}-{}.weights', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
print("Fitting...")
resnet.fit({
            'input':X_train,
            'output':Y_train
           },
           batch_size=batch_size,
           nb_epoch=epochs,
           shuffle=True,
           show_accuracy=True,
           validation_data={
                            'input': X_test,
                            'output': Y_test
                            },
           callbacks=[cb, cb2])
