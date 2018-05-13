#
#  Used to test models of keras
#  date: 2018-5-13
#  a   : zhonghy
#

##"""Sequential model"""
###define model
##from keras.models import Sequential
##from keras.layers import Dense, Activation
##model = Sequential()
##model.add(Dense(output_dim=64, input_dim=100))
##model.add(Activation("relu"))
##model.add(Dense(output_dim=10))
##model.add(Activation("softmax"))
##
###compile model
##model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
##
###train and estimate model
##model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)
##loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)


"""Use keras to implement CNN"""
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('E:data\MNIST_data')
batch_size = 128
nb_classes = 10
nb_epoch = 12

#input image dimension
img_rows, img_cols = 28, 28
#number of filters
nb_filters = 32
pool_size = (2, 2)
kernel_size = (3, 3)

##(X_train, y_train), (X_test, y_test) = mnist.load_data()
##(X_train, y_train) = mnist.train()
##(X_test, y_test) = mnist.test()
X_train, y_train, X_test, y_test = mnist.train.images, mnist.train.labels,mnist.test.images, mnist.test.labels
                             


if K.image_dim_ordering() == 'th':
    #use order of Theano:(conv_dim1, channels, conv_dim2, conv_dim3)
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    #use order of TensorFlow:(conv_dim1, conv_dim2, conv_dim3, channels)
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
        
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

#transform vector to bit matrix
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#construct model, 2 conv, 1 pool, 2 fc(full connect)
model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

#compile model, use mutil categorical loss function and Adadelta optimizer
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

#use model.fit() function
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))

#evaluate model and calculate loss and accuracy
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])






























