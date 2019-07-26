
import tensorflow as tf
import numpy as np

x = [[1, 1, 1],
     [1, 1, 1]]
print(tf.reduce_sum(x))
print(tf.reduce_sum(x, 0))
print(tf.reduce_sum(x, 1))
print(tf.reduce_sum(x, 1, keep_dims=True))
print(tf.reduce_sum(x, [0, 1]))
