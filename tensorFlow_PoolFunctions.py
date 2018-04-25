#
#
#  a:zhonghy
#
#
#============================================


import tensorflow as tf
import numpy as np

""" avg_pool """
input_data = tf.Variable( np.random.rand(10, 6, 6, 3), dtype = np.float32)
filter_data = tf.Variable( np.random.rand(2, 2, 3, 10), dtype = np.float32) #2 x 2

y = tf.nn.conv2d(input_data, filter_data, strides = [1, 1, 1,1 ], padding = 'SAME')
output = tf.nn.avg_pool(value = y, ksize = [1, 2, 2, 1], strides = [1, 1, 1,1],
                        padding = 'SAME')
print(output.get_shape())



