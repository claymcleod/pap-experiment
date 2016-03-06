from __future__ import print_function
import argparse

parser = argparse.ArgumentParser(description='CIFAR10 DeepCNet script for PAP experiment')
parser.add_argument('activation', type=str, help='activation function')
parser.add_argument('-l', '--learningrate',default=0.001,
                    type=float, help='learning rate')
parser.add_argument('-b', '--batchsize', default=128,
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
print("| Dataset: CIFAR10 (DeepCNet_small)")
print("| Activation: {}".format(activation))
print("| Learning rate: {}".format(learning_rate))
print("| Batch size: {}".format(batch_size))
print("| Epochs: {}".format(epochs))
print('| Data augmentation: {}'.format(data_augmentation))
print('\\==========================/')
print()

X_train, X_test, Y_train, Y_test = util.get_cifar10()
dcn = util.build_deepcnet(5, 50, activation, final_c1=True)
util.compile_deepcnet(dcn, learning_rate)

cb = util.PersistentHistory('./cifar10-deepcnet_small-{}-{}.csv'.format(activation, learning_rate))

for i in range(0, epochs):
    print("Iteration {} of {}".format(i, epochs))
    dcn.fit(X_train, Y_train,
            batch_size=batch_size, nb_epoch=1,
            show_accuracy=True,
            shuffle=True,
            validation_data=(X_test, Y_test),
            callbacks=[cb])
    if i>5:
        dcn.optimizer.lr.set_value(0.01)
        dcn.optimizer.decay.set_value(1e-5)
        print("Changed lr and weight decay")
