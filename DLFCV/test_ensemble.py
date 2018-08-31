#
# To construct and evaluate our ensemble of CNNs
# a : zhonghy
# date: 2018-8-27
#

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.datasets import cifar10
import numpy as np
import argparse
import glob
import os

# construct the argument and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-m", "--models",required=True,
                help="path to models directory")
args = vars(ap.parse_args())

# load the training and testing data, then scale it into the
# range [0, 1]
(testX, testY) = cifar10.load_data()[1]
testX = testX.astype("float") / 255.0

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

# convert the labels from integers to vectors
lb = LabelBinarizer()
testY = lb.fit_transform(testY)

# construct the path used to collect the models then initialize the
# models list
modelsPath = os.path.sep.join([args["models"], "*.model"])
modelsPath = list(glob.glob(modelsPath))
models = []

# load over the model paths, loading the model, and adding it to
# the list of models
for (i, modelPath) in enumerate(modelsPath):
    print("[INFO] loading model{}/{}".format(i + 1,
                                             len(modelsPath)))
    models.append(load_model(modelPath))

# initialize the list of predictions
print("[INFO] evaluating ensemble...")
predictions = []

# loop over the models
for model in models:
    # use current model to predicting and store
    predictions.append(model.predict(testX, batch_size=64))

# average the probabilities across all model predictions, then show
# a classification report
predictions = np.average(predictions, axis=0)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=labelNames))

