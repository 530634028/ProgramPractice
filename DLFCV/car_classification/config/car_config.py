"""
 config file for cars classification
 a : zhonghy
 date: 2018-1-8

"""
# import the necessary packages
from os import path

# define the base path to where the ImageNet dataset
# devkit are stored on disk
BASE_PATH = "../../../data/cars"

# based on the base path, derive the images base path, image sets
# path, and devkit path
IMAGES_PATH = path.sep.join([BASE_PATH, "cars"])
LABELS_PATH = path.sep.join([BASE_PATH, "complete_dataset.csv"])

# define the path to the output training, validation, and testing
# lists
MX_OUTPUT = BASE_PATH
TRAIN_MX_LIST = path.sep.join([MX_OUTPUT, "lists/train.lst"])
VAL_MX_LIST = path.sep.join([MX_OUTPUT, "lists/val.lst"])
TEST_MX_LIST = path.sep.join([MX_OUTPUT, "lists/test.lst"])

# define the path to the output training, validation, and testing
# image records
TRAIN_MX_REC = path.sep.join([MX_OUTPUT, "rec/train.rec"])
VAL_MX_REC = path.sep.join([MX_OUTPUT, "rec/val.rec"])
TEST_MX_REC = path.sep.join([MX_OUTPUT, "rec/test.rec"])

# define the path to the label encoder
LABEL_ENCODER_PATH = path.sep.join([BASE_PATH, "output/le.cpickle"])


# define the RGB means from the ImageNet dataset
R_MEAN = 123.68
G_MEAN = 116.779
B_MEAN = 103.939

# define the percentage of validation and testing images relative
# to the number of training images
NUM_CLASSES = 164
NUM_VAL_IMAGES = 0.16
NUM_TEST_IMAGES = 0.15

# define the batch size and number of devices used for training
BATCH_SIZE = 32
NUM_DEVICES = 1


