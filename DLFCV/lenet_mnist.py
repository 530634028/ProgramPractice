#
# Train letnet on MNIST dataset with keras
# a: zhonghy
# date: 2018-7-23
#
#


# import the necessary packages
from tensorflow.examples.tutorials.mnist import input_data

from pyimagesearch.nn.conv.lenet import LeNet
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np


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

# if we are using "channels first" ordering, then reshape the
# design matrix such that the matrix is:
# num_samples x depth x rows x columns
if K.image_data_format() == "channels_first":
    trainX = trainX.reshape(trainX.shape[0], 1, 28, 28)
    testX = testX.reshape(testX.shape[0], 1, 28, 28)
else:
    trainX = trainX.reshape(trainX.shape[0], 28, 28, 1)
    testX = testX.reshape(testX.shape[0],  28, 28, 1)

# convert the labels from integers to vectors
le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)

# intialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=128, epochs=20, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=[str(x) for x in le.classes_]))

# plot the training loss and accuarcy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuarcy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuarcy")
plt.legend()
plt.show()









