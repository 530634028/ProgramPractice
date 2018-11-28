#
#
#  date:2018-4-28
#  a   : zhonghy
#
#


#Load models

from google.protobuf import text_format

import tensorflow as tf
import numpy as np
import os

v = tf.Variable(0, name='my_variable')
sess = tf.Session()
tf.train.write_graph(sess.graph_def, 'F:/data/tfmodel', 'train.pbtxt') #add as_text=False

with tf.Session() as _sess:
    with tf.gfile.FastGFile("F:/data/tfmodel/train.pbtxt", 'rb') as f:
        graph_def = tf.GraphDef()
        
        text_format.Merge(f.read(), graph_def)
        
        graph_def.ParseFromString(f.read())
        _sess.graph.as_default()
        tf.import_graph_def(graph_def, name='tfgraph')
