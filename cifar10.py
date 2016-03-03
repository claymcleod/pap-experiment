from __future__ import print_function
import argparse
from keras.preprocessing.image import ImageDataGenerator

parser = argparse.ArgumentParser(description='CIFAR10 script for PAP experiment')
parser.add_argument('activation', type=str, help='activation function')
parser.add_argument('-l', '--learningrate',default=0.1,
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
print('/====================\\')
print("| Dataset: CIFAR10")
print("| Activation: {}".format(activation))
print("| Learning rate: {}".format(learning_rate))
print("| Batch size: {}".format(batch_size))
print("| Epochs: {}".format(epochs))
print('| Data augmentation: {}'.format(data_augmentation))
print('\\====================/')
print()

X_train, X_test, Y_train, Y_test = util.get_cifar10()
model = util.get_cifar10_model(activation, learning_rate)

if not data_augmentation:
    model.fit(X_train, Y_train,
              batch_size=batch_size, nb_epoch=epochs,
              show_accuracy=True,
              shuffle=True,
              validation_data=(X_test, Y_test),
              callbacks=[util.PersistentHistory('./cifar10-{}-{}.csv'.format(activation, learning_rate))])
else:
    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=epochs, show_accuracy=True,
                        validation_data=(X_test, Y_test),
                        nb_worker=5,
                        callbacks=[util.PersistentHistory('./cifar10-{}-{}.csv'.format(activation, learning_rate))])
