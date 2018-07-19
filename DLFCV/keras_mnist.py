#
# Train our standard FC(fully-connected) layer networks
# implementation on MNIST dataset with keras
# (implement feedforward, multi-layer networks and
#  apply them to the MNIST)
# a: zhonghy
# date: 2018-7-12
#
#

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import argparse

from tensorflow.examples.tutorials.mnist import input_data

### construct the argument parse and parse the arguments
##ap = argparse.ArgumentParser()
##ap.add_argument("-o", "--output", required=True,
##                help="path to the output loss/accuracy plot")
##args = vars(ap.parse_args())

# grab the MNIST dataset (if this is your first time running this
# script, the download may take a minute -- the 55MB MNIST dataset
# will be downloaded)
print("[INFO] loading MNIST (full) dataset...")
dataset = input_data.read_data_sets("F:\\data\\MNIST_data", one_hot=True)
#dataset = datasets.fetch_mldata("F:\\data\\MNIST_data") #("MNIST Original")

# scale the raw pixel intensities to the range [0, 1.0], then
# construct the training and testing splits
##data = dataset.astype("float") / 255.0
##(trainX, testX, trainY, testY) = train_test_split(data,
##                                                  dataset.target, test_size=0.25)
trainX, trainY = dataset.train.images, dataset.train.labels
testX, testY = dataset.test.images, dataset.test.labels
trainX = trainX.astype("float") / 255.0
#trainY = trainY.astype("float") / 255.0
testX = testX.astype("float") / 255.0
#testY = testY.astype("float") / 255.0


# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# define the 784-256-128-10 architecture using keras
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

# train the model using SGD
print("[INFO] training network...")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
              metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              epochs=100, batch_size=128)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=[str(x) for x in lb.classes_]))

# plot the training loss and accuarcy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuarcy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuarcy")
plt.legend()
plt.show()
plt.savefig("F:\\data")  # args["output"]











