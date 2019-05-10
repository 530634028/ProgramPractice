
"""
#
#  Used to test model saving of keras
#  data: 2018-5-14
#  a   : zhonghy
#

"""

import tensorflow as tf
import numpy as np
import os
from keras import backend as K
from keras.layers import Dense, Activation, Flatten, Dropout, RepeatVector, TimeDistributed
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import h5py
import tempfile
from numpy.testing import assert_allclose
from numpy.testing import assert_raises

from keras.models import Model, Sequential
from keras.layers import Lambda, Bidirectional, GRU, LSTM, CuDNNGRU, CuDNNLSTM
from keras.layers import Conv2D, Flatten
from keras.layers import Input, InputLayer
from keras import optimizers
from keras import losses
from keras import objectives
from keras import metrics

from keras.utils.test_utils import keras_test
from keras.models import save_model, load_model
from keras.models import model_from_json
from keras.models import model_from_yaml

##def test_squential_model_saving():
model = Sequential()
model.add(Dense(2, input_dim=3))
model.add(RepeatVector(3))
model.add(TimeDistributed(Dense(3)))
model.compile(loss=objectives.MSE,
              optimizer=optimizers.RMSprop(lr=0.0001),
              metrics=[metrics.categorical_accuracy],
              sample_weight_mode='temporal')
x = np.random.random((1, 3))
y = np.random.random((1, 3, 3))
model.train_on_batch(x, y)

out = model.predict(x)
_, fname = tempfile.mkstemp('.h5') #create h5 file why can't set path?
#tempfile already set path????
save_model(model, fname)

new_model = load_model(fname)
##    os.remove(fname)

out2 = new_model.predict(x)
assert_allclose(out, out2, atol=1e-05)

#check old and new is same or not (identical)
x = np.random.random((1, 3))
y = np.random.random((1, 3, 3))
model.train_on_batch(x, y)

new_model.train_on_batch(x, y)
#wrong write new_model_train_on_batch
#get error:Blas GEMM??
out = model.predict(x)
out2 = new_model.predict(x)
assert_allclose(out, out2, atol=1e-05)
##test_squential_model_saving()





"""json and yaml file"""
##json_string = model.to_json()
yaml_string = model.to_yaml()

##model = model_from_json(json_string)
model = model_from_yaml(yaml_string)

model.save_weights('my_model_weights.h5')
model.load_weights('my_model_weights.h5')











    
