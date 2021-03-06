#
# Use RNN to train mnist
# date:2018-5-15
# a   :zhonghy
#
#

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


"""input data"""
mnist = input_data.read_data_sets("F:\data\MNIST_data", one_hot=True)
##trX, trY = mnist.train.images, mnist.train.labels
##teX, teY = mnist.test.images, mnist.test.labels

#set training super parameters
lr = 0.001
training_iters = 100000
batch_size = 128

#parameters of RNN
n_inputs = 28 # num of input layer
n_steps = 28 # 28
n_hidden_units = 128 # number of neure of hidden layer
n_classes = 10 # output number

#define input data and placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

#define weights
weights = {
    #(28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    #(128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
    }
biases = {
    #(128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    #(10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
    }

#define RNN
def RNN(X, weights, biases):

    #transform X to X(128(batch)*28(steps)*28(inputs))
    X = tf.reshape(X, [-1, n_inputs])

    #hidden layer
    #X_in = (128*28*128)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    #X_in ==> (128 28 128)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    
    #use basic LSTM Cell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0,
                                             state_is_tuple=True)
    #initialize to zero, lstm uint(c_state,h_state)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    #dynamic_rnn recepte tensor (batch, steps, inputs) as X_in
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=
                                             init_state, time_major=False)
    results = tf.matmul(final_state[1], weights['out'])
    return results

#define loss function and optimizer
pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

#define calculate method for predict result and accuracy
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#train and estimate model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
                x: batch_xs,
                y: batch_ys,
                }))
        step += 1

