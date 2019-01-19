"""
 configure file of ssds and rcnn -- traffic signs detector
 a : zhonghy
 date: 2019-1-18

"""
# import the necessary packages
import os

# inialize the base path for the LISA dataset
BASE_PATH = "lisa"

# build the path to the annotations file
ANNOT_PATH = os.path.sep.join([BASE_PATH, "allAnnotations.csv"])

# define the path to the output training, validation, and testing
# image records
TRAIN_RECORD = os.path.sep.join([BASE_PATH, "records/training.record"])
TEST_RECORD = path.sep.join([BASE_PATH, "records/testing.record"])
CLASSES_FILE = path.sep.join([BASE_PATH, "records/classes.record"])

# initialize the test split size
TEST_SIZE = 0.25

# initialize the class labels dictionary
CLASSES = {"pedestrianCrossing": 1, "signalAhead": 2, "stop": 3}

