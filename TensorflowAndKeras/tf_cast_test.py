
import tensorflow as tf
x = tf.constant([1.8, 2.2], dtype=tf.float32)
y = tf.cast(x, tf.int32)
elems = tf.map_fn(lambda y: y, y, dtype=tf.int32)
print(elems[0])

##for i in range(0, 1):
##   print(y[i])
