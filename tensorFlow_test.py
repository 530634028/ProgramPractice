# 
#
#  a:zhonghy
# 
# 
# ==============================================================================

"""test for sigmoid() function."""
import tensorflow as tf
a = tf.constant([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
sess = tf.Session()
print(sess.run(tf.sigmoid(a)))
