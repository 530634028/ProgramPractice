

import numpy as np
import tensorflow as tf

ndarray = np.ones([3, 3])

tensor = tf.multiply(ndarray, 42)
print(tensor)