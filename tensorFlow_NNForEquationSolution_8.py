#
# NN for equation solution
# date: 2018-5-11
# a   : zhonghy
#

import tensorflow as tf
import numpy as np


"""construct net"""  """function defied before"""
def add_layer(inputs, in_size, out_size, activation_function=None):
    #construct weights: in_size x out_size matrix
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    #construct biases: 1 x out_size matrix
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    #mat mul
    Wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs #obtain outputs

"""Load data"""
#crate quadric equation with one variable
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape) #noise
y_data = np.square(x_data) - 0.5 + noise #y = x^2 -0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
    
#construct hidden layer that consist 10 neure
h1 = add_layer(xs, 1, 20, activation_function=tf.nn.relu)
#construct output layer that consist 1 neure
prediction = add_layer(h1, 20, 1, activation_function=None)

"""construct loss function"""
#calculate error between prediction and reality
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

"""train model"""
init = tf.global_variables_initializer() #initialize all variables
sess = tf.Session()
sess.run(init)

for i in range(1000): #train 1000 times
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))











    
            
