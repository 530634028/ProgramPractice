#
# Use AutoEncoder to train mnist
# date:2018-5-15
# a   :zhonghy
#
#
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt #show images

from tensorflow.examples.tutorials.mnist import input_data


"""input data"""
mnist = input_data.read_data_sets("F:\data\MNIST_data", one_hot=True)
##trX, trY = mnist.train.images, mnist.train.labels

"""set supter parameters for training"""
learning_rate = 0.01
training_epochs = 40 #20
batch_size = 256
display_step = 1

#number of image for testing
examples_to_show = 10

#parameters of net
n_hidden_1 = 256 #number of neures in layer 1
n_hidden_2 = 128 #number of neures in layer 2
n_input = 784    #number of eigen value: 28 x 28 = 784

X = tf.placeholder("float", [None, n_input])

#initialize weights and biases in each layer
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
    }
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
    }

"""define encoder and decoder function"""
#define encoder
def encoder(x):
    #Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    #Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2

#define decoder
def decoder(x):
    #Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    #Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

#construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

#obtain predict value
y_pred = decoder_op
#obtain real value, namely input value
y_true = X

"""define loss function and optimizer"""
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

"""train and estimate model"""
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    
    #begin training
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #Run optimizer op (backprop) and cost op(to ge loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        #each epoch print loss value
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    #test in test dataset
    encode_decode = sess.run(y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    #compare origin images between images established by encoder
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28))) #test data
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28))) #establih data
    f.show()
    plt.draw()
    plt.waitforbuttonpress()





