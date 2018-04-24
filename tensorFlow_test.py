# 
#
#  a:zhonghy
# 
# 
# ==============================================================================

import tensorflow as tf
import numpy as np

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



##"""test for conv2d() function."""
##input_data = tf.Variable( np.random.rand(10,9,9,3), dtype = np.float32 )
##filter_data = tf.Variable( np.random.rand(2, 2, 3, 2), dtype = np.float32 )
##y = tf.nn.conv2d(input_data, filter_data, strides = [1, 1, 1, 1], padding = 'SAME')
##print(tf.shape(y))


"""test for depthwise_conv2d() function."""
input_data = tf.Variable( np.random.rand(10, 9, 9, 3), dtype = np.float32 )
filter_data = tf.Variable( np.random.rand(2, 2, 3, 5), dtype = np.float32 )
y = tf.nn.depthwise_conv2d(input_data, filter_data, strides = [1, 1, 1, 1], padding = 'SAME')
print(tf.shape(y))













    
