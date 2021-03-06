from __future__ import print_function
import util
import argparse

parser = argparse.ArgumentParser(description='MNIST script for PAP experiment')
parser.add_argument('activation', type=str, help='activation function')
parser.add_argument('-l', '--learningrate',default=0.01,
                    type=float, help='learning rate')
parser.add_argument('-b', '--batchsize', default=32,
                    type=int, help='batch size')
parser.add_argument('-e', '--epochs', default=100,
                    type=int, help='epochs')
args = parser.parse_args()

activation = args.activation
learning_rate = args.learningrate
batch_size = args.batchsize
epochs = args.epochs
initialization = 'uniform'


print()
print('/====================\\')
print("| Dataset: MNIST")
print("| Activation: {}".format(activation))
print("| Learning rate: {}".format(learning_rate))
print("| Batch size: {}".format(batch_size))
print("| Epochs: {}".format(epochs))
print("| Initialization: {}".format(initialization))
print('\\====================/')
print()

X_train, X_test, Y_train, Y_test = util.get_mnist()
model = util.get_mnist_model(activation, initialization, learning_rate)

model.fit(X_train, Y_train,
          batch_size=batch_size, nb_epoch=epochs,
          show_accuracy=True, verbose=2,
          validation_data=(X_test, Y_test),
          callbacks=[util.PersistentHistory('./mnist-{}-{}.csv'.format(activation, learning_rate))])
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
