# 
#
#  a:zhonghy
# 
# 
# ==============================================================================

import tensorflow as tf
import numpy as np


###################Active Function###############################################
##"""test for sigmoid() function."""
##a = tf.constant([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
##sess = tf.Session()
##print(sess.run(tf.sigmoid(a)))



##"""test for relu() function."""
##a = tf.constant([-1.0, 2.0])
##with tf.Session() as sess:
##    b = tf.nn.relu(a)
##    print(sess.run(b))



##"""test for dropout() function."""
##a = tf.constant([[-1.0, 2.0, 3.0, 4.0]])
##with tf.Session() as sess:
##    b = tf.nn.dropout(a, 0.5, noise_shape = [1,4])
##    print(sess.run(b))
##    b = tf.nn.dropout(a, 0.5, noise_shape = [1,1])
##    print(sess.run(b))


###################Convolution Functions######################################### 

##"""test for conv2d() function."""
##input_data = tf.Variable( np.random.rand(10,9,9,3), dtype = np.float32 )
##filter_data = tf.Variable( np.random.rand(2, 2, 3, 2), dtype = np.float32 )
##y = tf.nn.conv2d(input_data, filter_data, strides = [1, 1, 1, 1], padding = 'SAME')
##print(y.get_shape())
##init = tf.global_variables_initializer()
##with tf.Session() as sess:
##    sess.run(init)
##    print(sess.run(y))   #wrong write sess as see? why didn't print result?

##input = tf.Variable(tf.random_normal([1,3,3,5]))  
##filter = tf.Variable(tf.random_normal([1,1,5,1]))  
##op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
##init = tf.global_variables_initializer()
##with tf.Session() as sess:
##    sess.run(init)
##    print(sess.run(op2))   #wrong write sess as see?



##"""test for depthwise_conv2d() function."""
##input_data = tf.Variable( np.random.rand(10, 9, 9, 3), dtype = np.float32 )
##filter_data = tf.Variable( np.random.rand(2, 2, 3, 5), dtype = np.float32 )
##y = tf.nn.depthwise_conv2d(input_data, filter_data, strides = [1, 1, 1, 1], padding = 'SAME')
##print(y.get_shape())


##"""test for separable_conv2d() function."""
##input_data = tf.Variable( np.random.rand(10, 9, 9, 3), dtype = np.float32)
##depthwise_filter = tf.Variable( np.random.rand(2, 2, 3, 5), dtype = np.float32)
##pointwise_filter = tf.Variable( np.random.rand(1, 1, 15, 20), dtype = np.float32)
###out_channels >= channel_multiplier * inchannels
##y = tf.nn.separable_conv2d(input_data, depthwise_filter, pointwise_filter,
##                           strides = [1, 1, 1, 1], padding = 'SAME')
##print(y.get_shape())


##"""test for atrous_conv2d() function."""
##input_data = tf.Variable( np.random.rand(1, 5, 5, 1), dtype = np.float32)
##filters = tf.Variable( np.random.rand(3, 3, 1, 1), dtype = np.float32)
##y = tf.nn.atrous_conv2d(input_data, filters, 2, padding = 'SAME')
##print(y.get_shape())

##"""test for conv2d_transpose() function."""
##x = tf.random_normal(shape = [1, 3, 3, 1])
##kernel = tf.random_normal(shape = [2, 2, 3, 1])
##y = tf.nn.conv2d_transpose(x, kernel, output_shape = [1, 5, 5, 3],
##                           strides = [1, 2, 2, 1], padding = 'SAME')
##print(y.get_shape())
##init = tf.global_variables_initializer()
##with tf.Session() as sess:
##    sess.run(init)
##    print(sess.run(y))





    
