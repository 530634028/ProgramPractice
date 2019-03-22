
"""
 Configure file of emotion recognition
 a : zhonghy
 date: 2018-12-5

"""


# import the necessary packages
from os import path

# define the base path to where the ImageNet dataset
# devkit are stored on disk
BASE_PATH = "F:/data/fer2013"

# use the base path to define the path to the input emotions file
INPUT_PATH = path.sep.join([BASE_PATH, "fer2013/fer2013.csv"])

# since we do not have access to testing data we need to
# take a number of images from the training data and use it instead
# NUM_CLASSES = 7
NUM_CLASSES = 6

# define the path to the output training, validation, and testing
# lists
TRAIN_HDF5 = path.sep.join([BASE_PATH, "hdf5/train.hdf5"])
VAL_HDF5 = path.sep.join([BASE_PATH, "hdf5/val.hdf5"])
TEST_HDF5 = path.sep.join([BASE_PATH, "hdf5/test.hdf5"])

# define the batch size and number of devices used for training
BATCH_SIZE = 128

# define the path to where output logs will be stored
OUTPUT_PATH = path.sep.join([BASE_PATH, "output"])


