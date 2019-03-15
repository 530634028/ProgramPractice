"""
 test predictions for age and gender classification
 a : zhonghy
 date: 2019-1-16

"""
# import the necessary packages
import  cv2

import sys
sys.path.append("F:\ProgramPractice\DLFCV")

from config import age_gender_deploy as deploy
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.meanpreprocessor import MeanPreprocessor
from pyimagesearch.preprocessing.croppreprocessor import CropPreprocessor
from pyimagesearch.utils.agegemderhelper import AgeGenderHelper
from imutils.face_utils.facealigner import FaceAligner
from imutils import face_utils
from imutils import paths
import numpy as np
import mxnet as mx
import argparse
import pickle
import imutils
import json
# test zhong
# import dlib
import os


# construct the argument and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image (or directory)")
args = vars(ap.parse_args())

# load the label encoder, followed by the testing dataset file
# then sample the testing set
print("[INFO] loading label encoders and mean files...")
ageLE = pickle.loads(open(deploy.AGE_LABEL_ENCODER, "rb").read())
genderLE = pickle.loads(open(deploy.GENDER_LABEL_ENCODER, "rb").read())
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
sp = SimplePreprocessor(width=256, height=256, inter=cv2.INTER_CUBIC)
cp = CropPreprocessor(width=227, height=227, horiz=True)
ageMp = MeanPreprocessor(ageMeans["R"], ageMeans["G"], ageMeans["B"])
genderMp = MeanPreprocessor(genderMeans["R"], genderMeans["G"], genderMeans["B"])
iap = ImageToArrayPreprocessor(dataFormat="channels_first")  # because mxnet use this channels

# initialize dlib's face detector (HOG-based), then create the
# the facial labdmark predictor and face aligner
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(deploy.DLIB_LANDMARK_PATH)
# fa = FaceAligner(predictor)

# initialize the list of image paths as just a single image
imagePaths = [args["image"]]

# if the input path is actually a directory, then list all image
# paths in then directory
if os.path.isdir(args["image"]):
    imagePaths = sorted(list(paths.list_files(args["image"])))

# loop over the image paths
for imagePath in imagePaths:
    # load the image from disk, resize it, and convert it to
    # grayscale
    print("[INFO] processing {}".format(imagePath))
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=800)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # test zhong
    # gray = image

    # detect faces in the grayscale image
    # rects = detector(gray, 1)
    # test zhong
    # rects = gray

    # determine the facial landmarks for the face region, then
    # align the face
    # shape = predictor(gray, rect)
    # face = fa.align(image, gray, rect)
    # test zhong
    shape = image.shape
    # face = rect

    # resize the face to a fixed size, then extract 10-crop
    # patches from it
    face = sp.preprocess(image)
    patches = cp.preprocess(image)

    # allocate memory for the age and gender pathces
    agePatches = np.zeros((patches.shape[0], 3, 227, 227),
                          dtype="float")
    genderPatches = np.zeros((patches.shape[0], 3, 227, 227),
                          dtype="float")

    # print(patches.shape[0])

    # loop over the patches
    for j in np.arange(0, patches.shape[0]):
        # perform mean subtraction on the patch
        # agePatch = ageMp.preprocess(patches[j])
        # genderPatch = genderMp.preprocess(patches[j])
        # agePatch = iap.preprocess(agePatch)
        # genderPatch = iap.preprocess(genderPatch)

        # test zhong
        agePatch = iap.preprocess(patches[j])
        genderPatch = iap.preprocess(patches[j])


        # update the respective patches lists
        agePatches[j] = agePatch
        genderPatches[j] = genderPatch

    # classify the image and grab the indexes of the top-5 predictions
    # face = cp.preprocess(face)
    # face = iap.preprocess(face)
    #
    agePreds = ageModel.predict(agePatches)
    genderPreds = genderModel.predict(genderPatches)

    # compute the average for each class label based on the
    # predictions for the patches
    agePreds = agePreds.mean(axis=0)
    genderPreds = genderPreds.mean(axis=0)

    # visualize the age and gender predictions
    # ageCanvas = AgeGenderHelper.visualizeAge(agePreds, ageLE)  # we didn't have this functions
    # genderCanvas = AgeGenderHelper.visualizeAge(genderPreds,
    #                                             genderLE)

    agh = AgeGenderHelper(deploy)
    # print(ageLabel)
    ageCanvas = agh.visualizeAge(agePreds, ageLE)
    genderCanvas = agh.visualizeGender(genderPreds, genderLE)

    # draw the bounding box around the face
    clone = image.copy()
    # (x, y, w, h) = face_utils.rect_to_bb(rect)
    # cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0),
    #               2)
    # test zhong
    text = "Actual: {}-{}"
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 3)

    # show the image
    # cv2.imshow("Input", clone)
    cv2.imshow("Face", face)
    cv2.imshow("Age Probabilities", ageCanvas)
    cv2.imshow("Gender Probabilities", genderCanvas)
    cv2.waitKey(0)


# python test_prediction.py --image examples