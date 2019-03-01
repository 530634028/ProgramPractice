"""
 configure file of ssds and rcnn(single shot)
 -- traffic signs detector
 a : zhonghy
 date: 2019-1-24

"""
# import the necessary packages
import os

# inialize the base path for the LISA dataset
BASE_PATH = "F:\data\dlib_front_and_rear_vehicles_v1"  # need to use real path

# build the path to the training and testing XML files
TRAIN_XML = os.path.sep.join([BASE_PATH, "training.xml"])
TEST_XML = os.path.sep.join([BASE_PATH, "testing.xml"])

# define the path to the output training, validation, and testing
# image records
TRAIN_RECORD = os.path.sep.join([BASE_PATH, "records/training.record"])
TEST_RECORD = os.path.sep.join([BASE_PATH, "records/testing.record"])
CLASSES_FILE = os.path.sep.join([BASE_PATH, "records/classes.pbtxt"])

# initialize the class labels dictionary
CLASSES = {"rear": 1, "front": 2}


