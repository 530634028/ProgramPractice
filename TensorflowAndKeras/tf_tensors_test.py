

import numpy as np
import tensorflow as tf

# ndarray = np.ones([3, 3])
#
# tensor = tf.multiply(ndarray, 42)
# print(tensor)

sess = tf.Session()

x_vals = np.array([1., 3., 5., 7., 9.])
x_data = tf.placeholder(tf.float32)  # like variables of function
m_const = tf.constant(3.)
my_product = tf.multiply(x_data, m_const)

# with tf.Session() as sess:
for x_val in x_vals:
     print(sess.run(my_product, feed_dict={x_data:x_val}))

