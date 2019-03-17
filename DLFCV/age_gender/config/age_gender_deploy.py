"""
 deployment file for age and gender classification
 a : zhonghy
 date: 2019-1-16

"""

# import the necessary packages
import sys
# sys.path.append("F:\ProgramPractice\DLFCV")
sys.path.append("E:\ZWDX_Learn\ProgramPractice\DLFCV")

from age_gender.config.age_gender_config import OUTPU_BASE
from os import path

# define the path to the dlib facial landmark predictor
DLIB_LANDMARK_PATH = "shape_predictor_68_face_landmarks.dat"

# defeine the path to the age network + supporting files
# company
# AGE_NETWORK_PATH = "../../../data/adience/checkpoints/age"
# home
AGE_NETWORK_PATH = "../../../../data/adience/checkpoints/age"

AGE_PREFIX = "agenet"
AGE_EPOCH = 150
AGE_LABEL_ENCODER = path.sep.join([OUTPU_BASE, "age_le.cpickle"])
AGE_MEANS = path.sep.join([OUTPU_BASE, "age_adience_mean.json"])

# defeine the path to the gender network + supporting files
# company
# GENDER_NETWORK_PATH = "../../../data/adience/checkpoints/gender"
# home
GENDER_NETWORK_PATH = "../../../../data/adience/checkpoints/gender"

GENDER_PREFIX = "gendernet"
GENDER_EPOCH = 110
GENDER_LABEL_ENCODER = path.sep.join([OUTPU_BASE, "gender_le.cpickle"])
GENDER_MEANS = path.sep.join([OUTPU_BASE, "gender_adience_mean.json"])


