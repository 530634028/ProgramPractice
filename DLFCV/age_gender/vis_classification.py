"""
 visualize classification for age and gender classification
 a : zhonghy
 date: 2019-1-16

"""

# due to mxnet seg-fault issue, need to place OpenCV import at the
# top of the file
import cv2

# import the necessary packages
import sys
sys.path.append("F:\ProgramPractice\DLFCV")

from config import age_gender_config as config
from config import age_gender_deploy as deploy
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.meanpreprocessor import MeanPreprocessor
from pyimagesearch.utils.agegemderhelper import AgeGenderHelper
import numpy as np
import mxnet as mx
import argparse
import pickle
import imutils
import json
import os

# construct the argument and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--sample-size", type=int, default=10,
                help="number of images sampled from dataset")
args = vars(ap.parse_args())

# load the label encoder, followed by the testing dataset file
# then sample the testing set
print("[INFO] loading label encoders and mean files...")
ageLE = pickle.loads(open(deploy.AGE_LABEL_ENCODER_PATH, "rb").read())
genderLE = pickle.loads(open(deploy.GENDER_LABEL_ENCODER_PATH, "rb").read())
ageMeans = json.loads(open(deploy.AGE_MEANS).read())
genderMeans = json.loads(open(deploy.GENDER_MEANS).read())

# load the models from disk
print("[INFO] loading pre-trained model...")
agePath = os.path.sep.join([deploy.AGE_NETWORK_PATH,
                                    deploy.AGE_PREFIX])
genderPath = os.path.sep.join([deploy.GENDER_NETWORK_PATH,
                                    deploy.GENDER_PREFIX])
ageModel = mx.model.FeedForward.load(
    agePath, deploy.AGE_EPOCH
)
genderModel = mx.model.FeedForward.load(
    genderPath, deploy.GENDER_EPOCH
)

# complie the model
print("[INFO] compiling models...")
ageModel = mx.model.FeedForward(
    ctx=[mx.gpu(0)],
    symbol=ageModel.symbol,
    arg_params=ageModel.arg_params,
    aux_params=ageModel.aux_params
)
genderModel = mx.model.FeedForward(
    ctx=[mx.gpu(0)],
    symbol=genderModel.symbol,
    arg_params=genderModel.arg_params,
    aux_params=genderModel.aux_params
)

# initialize the image pre-processors
sp = SimplePreprocessor(width=227, height=227, inter=cv2.INTER_CUBIC)
ageMp = MeanPreprocessor(ageMeans["R"], ageMeans["G"], ageMeans["B"])
genderMp = MeanPreprocessor(genderMeans["R"], genderMeans["G"], genderMeans["B"])
iap = ImageToArrayPreprocessor()

# then sample the testing set
rows = open(config.TEST_MX_LIST).read().strip().split("\n")
rows = np.random.choice(rows, size=args["sample_size"])

# loop over the testing images
for row in rows:
    # grab the target class label and the image path from the row
    (_, gtLabel, imagePath) = row.strip().split("\t")
    image = cv2.imread(imagePath)

    # pre-process images
    ageImage = iap.preprocess(ageMp.preprocess(sp.preprocess(image)))
    genderImage = iap.preprocess(genderMp.preprocess(sp.preprocess(image)))
    ageImage = np.expand_dims(ageImage, axis=0)
    genderImage = np.expand_dims(genderImage, axis=0)

    # classify the image and grab the indexes of the top-5 predictions
    agePreds = ageModel.predict(ageImage)[0]
    genderPreds = genderModel.predict(genderImage)[0]
    # sort teh predictions according to their probability
    ageIdxs = np.argsort(agePreds)[::-1]
    genderIdxs = np.argsort(genderPreds)[::-1]

    # visualize the age and gender predictions
    ageCanvas = AgeGenderHelper.visualizeAge(agePreds, ageLE) # we didn't have this functions
    genderCanvas = AgeGenderHelper.visualizeAge(genderPreds, genderLE)
    image = imutils.resize(image, width=400)

    # format and display the top predicted class label
    gtLabel = ageLE.inverse_transform(int(gtLabel))
    text = "Actual: {}-{}".format(*getLabel.split("_"))
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 3)

    # show the image
    cv2.imshow("Image", image)
    cv2.imshow("Age Probabilities", ageCanvas)
    cv2.imshow("Gender Probabilities", genderCanvas)
    cv2.waitKey(0)


