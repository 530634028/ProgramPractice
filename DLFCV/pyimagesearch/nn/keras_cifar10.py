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
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct the argument parse and parses the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# load the training and testing data, scale it int the range [0, 1]
# then reshape the design matrix
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
tainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
trainX = trainX.reshape((trainX.shape[0], 3072))
testX = testX.reshape((trainX.shape[0], 3072))





























