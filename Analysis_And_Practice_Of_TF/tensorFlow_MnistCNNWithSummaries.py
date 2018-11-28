#
# Use CNN to train mnist
# date:2018-5-15
# a   :zhonghy
#
#

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


"""input data"""
mnist = input_data.read_data_sets("F:\data\MNIST_data", one_hot=True)
trX, trY = mnist.train.images, mnist.train.labels
teX, teY = mnist.test.images, mnist.test.labels

trX = trX.reshape(-1, 28, 28, 1) #28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1) #28x28x1 input img

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

"""initilize werights"""
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

w = init_weights([3, 3, 1, 32])  #patch size 3x3, input dim 1, output dim 32
w2 = init_weights([3, 3, 32, 64])#patch size 3x3, input dim 32, output dim 64
w3 = init_weights([3, 3, 64, 128])#patch size 3x3, input dim 64, output dim 128
w4 = init_weights([128 * 4 * 4, 625])
#full connect, input dim 128x4x4 is ouput of poir layer(3D-1D), output dim 625
w_o = init_weights([625, 10]) #ouput layer,input dim 625, output dim 10(represent 10 labels)


"""define CNN model"""
#X:input data w:weights of each layer
#p_keep_conv, p_keep_hidden:dropout rate of nure remaining
def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    #1 conv and pool layer, dropout
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
    #l1a shape=(?, 28, 28, 32)
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #l1 shape=(?, 14, 14, 32)
    l1 = tf.nn.dropout(l1, p_keep_conv)

    #2 conv and pool layer, dropout
    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))
    #l2a shape=(?, 14, 14, 64)
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #l2 shape=(?, 7, 7, 64)
    l2 = tf.nn.dropout(l2, p_keep_conv)

    #3 conv and pool layer, dropout
    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME'))
    #l3a shape=(?, 7, 7, 128)
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #l3 shape=(?, 4, 4, 128)
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)

    #4 fully connect layer, dropout
    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    #5 ouput layer
    pyx = tf.matmul(l4, w_o)
    return pyx #return predict value


"""define placeholder and create net"""
p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

"""define loss function softmax_cross_entropy_with_logits and
   optimizer RMSProp"""
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

"""train and estimate model"""
batch_size = 128
test_size = 256  #256

#launch the graph in a session
with tf.Session() as sess:
    #you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(100):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX) + 1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})

        test_indices = np.arange(len(teX)) #Get A Test Batch, remember not trX
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]
        
        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0}))) 

