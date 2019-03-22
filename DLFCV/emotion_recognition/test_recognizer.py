

"""
#
# Test emotionnet on fer2013 dataset
# a : zhonghy
# date: 2018-10-29
#

"""

# import hte necessary packages
import sys
sys.path.append("F:\ProgramPractice\DLFCV")

from config import emotion_config as config
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.io.hdf5datasetgenerator import HDF5DatasetGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import argparse
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,
                help="path to model checkpoint to load")
args = vars(ap.parse_args())

testAug = ImageDataGenerator(rescale=1 / 255.0)
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
testGen = HDF5DatasetGenerator(config.TEST_HDF5, config.BATCH_SIZE,
                                aug=testAug, preprocessors=[iap],
                                classes=config.NUM_CLASSES)

print("[INFO] loading {}...".format(args["model"]))
fPath = os.path.sep.join([config.OUTPUT_PATH, args["model"]])
model = load_model(fPath)  # load_model(args["model"])


# train the network
(loss, acc) = model.evaluate_generator(
    testGen.generator(),
    steps=testGen.numImages // config.BATCH_SIZE,
    max_queue_size=config.BATCH_SIZE * 2,
)

# display the rank-1 and rank-5 accuracies
print("[INFO] accuracy: {:.2f}%".format(acc * 100))

# clost the testing database
testGen.close()

# python test_recognizer.py --model checkpoints/weights-050.hdf5
# [INFO] accuracy: 66.41%

