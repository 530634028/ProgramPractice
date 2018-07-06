#
# Implement of regularization for 'Animal' dataset
# a: zhonghy
# date: 2018-7-6
#
#

# import the necessary packages
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from DLFCV_pyimagesearch.preprocessing import SimplePreprocessor
from DLFCV_pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import argparse


