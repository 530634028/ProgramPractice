
#
# Implement for Mnist with Acgan
# date: 2018-5-21
# a   : zhonghy
#
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Flatten, Dropout, UpSampling2D, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt


""" from github"""
from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
from PIL import Image

from six.moves import range

from keras.datasets import mnist
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar

np.random.seed(1337)
K.set_image_dim_ordering('th')


#data input
##mnist = input_data.read_data_sets("F:\data\MNIST_data", one_hot=True);
##trX, trY = mnist.train.images, mnist.train.labels
##teX, teY = mnist.test.images, mnist.test.labels

""""""
def build_generator(latent_size):
    cnn = Sequential()

    cnn.add(Dense(1024, input_dim=latent_size, activation='relu'))
    cnn.add(Dense(128 * 7 * 7, activation='relu'))
    cnn.add(Reshape((128, 7, 7)))

    #up sample, image size be changed to 14 x 14 
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(256, 5, 5, border_mode='same',
                          activation='relu', init='glorot_normal'))

    #up sample, image size be changed to 28 x 28
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(128, 5, 5, border_mode='same',
                          activation='relu', init='glorot_normal'))

    #normal to one channel
    cnn.add(Convolution2D(1, 2, 2, border_mode='same',
                          activation='tanh', init='glorot_normal'))

    #generate input layer of model, eigen vectors
    latent = Input(shape=(latent_size, ))
    #generate input layer of model, labels
    image_class = Input(shape=(1,), dtype='int32')

    cls = Flatten()(Embedding(10, latent_size, init='glorot_normal')(image_class))
    h = merge([latent, cls], mode='mul')
##    h = layers.multiply([latent, cls])
    
    fake_image = cnn(h)  #output fake image
    return Model(input=[latent, image_class], output = fake_image)


def build_discriminator():
    #util Leaky ReLU
    cnn = Sequential()
    
    cnn.add(Convolution2D(32, 3, 3, border_mode='same', subsample=(2, 2),
                          input_shape=(1, 28, 28)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(128, 3, 3, border_mode='same', subsample=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(256, 3, 3, border_mode='same', subsample=(1, 1)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(1, 28, 28))

    features = cnn(image)

    #two ouput
    #ouput fake or right, range-0~1
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    #assit calssified discriminator, ouput images
    aux = Dense(10, activation='softmax', name='auxiliary')(features)

    return Model(input=image, output=[fake, aux])

#excute
if __name__ == '__main__':

    #define super parameters
    nb_epochs = 50
    batch_size = 100
    latent_size = 100

    #lr
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    #construct discriminator net
    discriminator = build_discriminator()
    discriminator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
        )

    #construct generator net
    generator = build_generator(latent_size)
    generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
                      loss='binary_crossentropy')

    latent = Input(shape=(latent_size, ))
    image_class = Input(shape=(1,), dtype='int32')

    #generate fake images
    fake = generator([latent, image_class])

    #generate combined model
    discriminator.trainable = False
    fake, aux = discriminator(fake)
    combined = Model(input=[latent, image_class], output=[fake, aux])

    combined.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
                     loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])
    
    #transform mnist to (..., 1, 28, 28), range in (-1, 1)
    mnist = input_data.read_data_sets("F:\data\MNIST_data");
    X_train, Y_train = mnist.train.images, mnist.train.labels
    X_test, Y_test = mnist.test.images, mnist.test.labels
##    (X_train, Y_train),(X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape([X_train.shape[0], 28, 28])
    X_test = X_test.reshape([X_test.shape[0], 28, 28])
    print(X_train.shape)
##    print(X_train[1])

    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=1)

    X_test = (X_test.astype(np.float32) - 127.5) /127.5
    X_test = np.expand_dims(X_test, axis=1)

    nb_train, nb_test = X_train.shape[0], X_test.shape[0]

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    for epoch in range(nb_epochs):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = int(X_train.shape[0] / batch_size)
        progress_bar = Progbar(target=nb_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in range(nb_batches):
            progress_bar.update(index)
            #generate one batch noise data
            noise = np.random.uniform(-1, 1, (batch_size, latent_size))

            print(X_train.shape)
            print(Y_train.shape)


            #obtain real data
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            label_batch = Y_train[index * batch_size:(index + 1) * batch_size]

            #generate some noise label
            sampled_labels = np.random.randint(0, 10, batch_size) #100 * 1

            #generate one batch fake images
            generated_images = generator.predict(
                [noise, sampled_labels.reshape((-1, 1))], verbose=0)
            
            print(image_batch.shape)
            print(generated_images.shape)

            X = np.concatenate((image_batch, generated_images))  
            y = np.array([1] * batch_size + [0] * batch_size)

            print(label_batch.shape)
            print(sampled_labels.shape)

            aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

            epoch_disc_loss.append(discriminator.train_on_batch(X, [y, aux_y]))

            #generate two batch noise and label
            noise = np.random.uniform(-1, 1, (2 * batch_size, latent_size))
            sampled_labels = np.random.randint(0, 10, 2 * batch_size)

            #train generator to trick the dicriminator, so set output to true
            trick = np.ones(2 * batch_size)

            epoch_gen_loss.append(combined.train_on_batch(
                [noise, sampled_labels.reshape((-1, 1))], [trick, sampled_labels]))

        print('\nTesting for epoch {}:'.format(epoch + 1))

        #evaluate test set
        #generate a new batch noise
        noise = np.random.uniform(-1, 1, (nb_test, latent_size))

        sampled_labels = np.random.randint(0, 10, nb_test)
        generated_images = generator.predict(
            [noise, sampled_labels.reshape((-1, 1))], verbose=False)
        
        X = np.concatenate((X_test, generated_images))
        y = np.array([1] * nb_test + [0] * nb_test)
        aux_y = np.concatenate((Y_test, sampled_labels), axis=0)

        #test discrimite or not
        discriminator_test_loss = discriminator.evaluate(X, [y, aux_y], verbose=False)
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis = 0)

        #generate two batch noise and label
        noise = np.random.uniform(-1, 1, (2 * nb_test, latent_size))
        sampled_labels = np.random.randint(0, 10, 2 * nb_test)

        #train generator to trick the dicriminator, so set output to true
        trick = np.ones(2 * nb_test)

        generator_test_loss = combined.evaluate(
            [noise, sampled_labels.reshape((-1, 1))],
            [trick, sampled_labels], verbose=False)

        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        #record loss value, then output
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format('component',
                                            *discriminator.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
        print(ROW_FMT.format('generator (train)', *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)', *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             *test_history['discriminator'][-1]))

        #save weights each epoch 
        generator.save_weights('params_generator_epoch_{0:03d}.hdf5'.format
                               (epoch), True)
        discriminator.save_weights('params_discriminator_epoch_{0:03d}.hdf5'.format
                               (epoch), True)

        #generate visual fake number for evolution
        noise = np.random.uniform(-1, 1, (100, latent_size))
        sampled_labels = np.array([[i] * 10 for i in range(10)]).reshape(-1, 1)
        generated_images = generator.predict(
                [noise, sampled_labels], verbose=0)

        #take into one grid
        img = (np.concatenate([r.reshape(-1, 28)
                               for r in np.split(generated_images, 10)
                               ], axis=-1) * 127.5 + 127.5).astype(np.uint8)
        Image.fromarray(img).save('plot_epoch_{0:03d}_generated.png'.format(epoch))
    pickle.dump({'train': train_history, 'test': test_history},
                open('F:\data\AcGanForMnist\acgan-history.pkl', 'wb'))


        

            






    





