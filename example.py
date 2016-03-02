import util
import theano.tensor as T
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense
from keras.optimizers import SGD

X_train, X_test, Y_train, Y_test = util.get_mnist()
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
#model.add(Activation('relu'))
model.add(util.ActivationPool([T.nnet.relu, util.step], threshold=True))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
#model.add(util.ActivationPool([T.nnet.relu, util.step], threshold=True))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))
sgd = SGD(lr=0.15, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(X_train, Y_train,
          batch_size=32, nb_epoch=200,
          show_accuracy=True, verbose=2,
          validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test,
                       show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
