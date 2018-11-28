#
#
# date : 2018-4-28
# a    : zhonghy
#
#

#load data
import tensorflow as tf
import numpy as np
import os

#input minst data from spcified dir
import tensorflow.examples.tutorials.mnist.input_data as input_data



mnist = input_data.read_data_sets("F:\\data\\MNIST_data", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
                     
X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

#define weight functions
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

#initilization weight of parameters
w_h = init_weights([784, 625])
w_h2 = init_weights([625, 625])
w_o = init_weights([625, 10])

###define weight functions
##def init_weights(shape):
##    return tf.Variable(tf.random_normal(shape, stddev=0.01))

#define model
def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
    #first full linked layer
    X = tf.nn.dropout(X, p_keep_input)
    h = tf.nn.relu(tf.matmul(X, w_h))

    h = tf.nn.dropout(h, p_keep_hidden)
    #second full linked layer
    h2 = tf.nn.relu(tf.matmul(h, w_h2))
    h2 = tf.nn.dropout(h2, p_keep_hidden)

    return tf.matmul(h2, w_o)  #input value of predict

#generate net model and obtain value of predict
p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

#define loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = py_x)) #right or not
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

#train model and save it
ckpt_dir = "./ckpt_dir"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

#counter for variable
global_step = tf.Variable(0, name='global_step', trainable= False)
saver = tf.train.Saver()

#variables after tf.train.Saver will be not saved
non_storable_variable = tf.Variable(777)

with tf.Session() as sess:
    tf.global_variables_initializer().run()   #new version function

    start = global_step.eval() #obtain value of global_step
    print("Start from:", start)

    for i in range(start, 100):
        #batch_size = 128
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                         p_keep_input: 0.8, p_keep_hidden: 0.5})

        global_step.assign(i).eval()   #update counter
        saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step) #save model


############Reload model#################
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path) #reload all parameters
        #start here for predict or train




                     


