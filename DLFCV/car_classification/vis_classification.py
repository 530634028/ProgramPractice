"""
 visualize vehicle classification result
 a : zhonghy
 date: 2018-1-9

"""
# due to mxnet seg-fault issue, need to place OpenCV import at the
# top of the file
import cv2

# import the necessary packages
import sys
# sys.path.append("F:\ProgramPractice\DLFCV")
sys.path.append("E:\ZWDX_Learn\ProgramPractice\DLFCV")

from config import car_config as config
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from pyimagesearch.preprocessing.meanpreprocessor import MeanPreprocessor
import numpy as np
import mxnet as mx
import argparse
import pickle
import imutils
import os

# construct the argument and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
                help="path to output checkpoint directory")
ap.add_argument("-p", "--prefix", required=True,
                help="name of model prefix")
ap.add_argument("-e", "--epoch", type=int, required=True,
                help="epoch # to load")
ap.add_argument("-s", "--sample-size", type=int, default=10,
                help="number of images sampled from dataset")
args = vars(ap.parse_args())

# load the label encoder, followed by the testing dataset file
# then sample the testing set
le = pickle.loads(open(config.LABEL_ENCODER_PATH, "rb").read())
rows = open(config.TEST_MX_LIST).read().strip().split("\n")
rows = np.random.choice(rows, size=args["sample_size"])

# build the VGG16 architecture
print("[INFO] loading pre-trained model...")
checkpointsPath = os.path.sep.join([args["checkpoints"],
                                    args["prefix"]])
model = mx.model.FeedForward.load(
    checkpointsPath, args["epoch"]
)

# complie the model
model = mx.model.FeedForward(
    ctx=[mx.gpu(0)],
    symbol=model.symbol,
    arg_params=model.arg_params,
    aux_params=model.aux_params
)

# initialize the image pre-processors
sp = AspectAwarePreprocessor(width=224, height=224)
mp = MeanPreprocessor(config.R_MEAN, config.G_MEAN, config.B_MEAN)
iap = ImageToArrayPreprocessor(dataFormat="channels_first")

# loop over the testing images
for row in rows:
    # grab the target class label and the image path from the row
    (target, imagePath) = row.split("\t")[1:]
    target = int(target)

    # load the image from disk and pre-process it by resizing the
    # image and applying the pre-processors
    image = cv2.imread(imagePath)
    orig = image.copy()
    orig = imutils.resize(orig, width=min(500, orig.shape[1]))
    image = iap.preprocess(mp.preprocess(sp.preprocess(image)))
    image = np.expand_dims(image, axis=0)

    # classify the image and grab the indexes of the top-5 predictions
    preds = model.predict(image)[0]
    idxs = np.argsort(preds)[::-1][:5]

    # show the true class label
    # classes = le.classes_
    # print(classes[target])
    print("[INFO] actual={}".format(le.inverse_transform([target])[0][:]))  # why can't use the inverse_transform function
                                                                      # inverse_transform function need list (--[]--) parameters  [0][:]
    # format and display the top predicted class label
    label = le.inverse_transform([idxs[0]])
    # label = label.replace(":", " ")
    # label = classes[idxs[0]]

    label = "{}: {:.2f}%".format(label[0][:], preds[idxs[0]] * 100)  # label
    cv2.putText(orig, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)

    # loop over the predictions and display them
    for (i, prob) in zip(idxs, preds):
        labelTmp = le.inverse_transform([i])
        labelT = labelTmp[0][:]
        print("\t[INFO] predicted={}, probability={:.2f}%".format(
            labelT, preds[i] * 100      # le.inverse_transform(i)
        ))

    # show the image
    cv2.imshow("Image", orig)
    cv2.waitKey()

