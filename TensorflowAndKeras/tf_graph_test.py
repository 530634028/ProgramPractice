
"""
  This program is used to test tf.Graph() function
  a : zhonghy
  date : 2019-7-25

"""

import tensorflow as tf

# 1. Using Graph.as_default():
g = tf.Graph()
with g.as_default():
    c = tf.constant(5.0)
    assert c.graph is g

# 2. Constructing and making default:
with tf.Graph().as_default() as g:
    c = tf.constant(5.0)
    assert c.graph is g




