import util

X_train, X_test, Y_train, Y_test = util.get_cifar10()
model = util.get_nonpap_model(3, 32, 32, 10, 'categorical_crossentropy', 'sgd')
model.fit({'input': X_train, 'output': Y_train},
          validation_data={'input': X_test, 'output': Y_test},
          batch_size=32, nb_epoch=500)
