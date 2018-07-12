#
# Train our standard FC(fully-connected) layer networks
# implementation on CIFAR dataset with keras
# a: zhonghy
# date: 2018-7-12
#
#

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras,layers.core import Dense
from keras.optimizers import SGD
from keras.datasets import cifar10
import tensorflow as tf
import matplotlib.
