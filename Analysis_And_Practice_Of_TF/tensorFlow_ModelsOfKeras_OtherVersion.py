#
#  Used to test models of keras with other version
#  date: 2018-5-14
#  a   : zhonghy 
#

"""Use keras to implement CNN"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from tensorflow.examples.tutorials.mnist import input_data
##from tensorflow.contrib.learn.python.learn.datasets.mnist import mnist
##from keras.datasets import mnist   #can use mnist.load_data() -- dowmload mnist
import os
import matplotlib.pyplot as plt #show images

mnist = input_data.read_data_sets('F:data\MNIST_data')
batch_size = 128
nb_classes = 10
nb_epoch = 12

#input image dimension
img_rows, img_cols = 28, 28
#number of filters
nb_filters = 32
pool_size = (2, 2)
kernel_size = (3, 3)

#input mnist data
##(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, y_train = mnist.train.images, mnist.train.labels
X_test, y_test = mnist.test.images, mnist.test.labels

X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)

# uint transform to float32 
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = (X_train - 127) / 127
X_test = (X_test - 127) / 127

nb_classes = 10
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Dense(512, input_shape=(784,), kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(Dropout(0.2)) 

model.add(Dense(512, kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(Dropout(0.2)) 

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1, validation_split=0.05)

loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Accuracy:', accuracy)




