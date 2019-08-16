

"""
  Test for reading csv file.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import steel_defect_detection_config as config

from matplotlib import pyplot as plt
import cv2
import csv
from os import path

train_csv_file = csv.reader(open(path.sep.join([config.BASE_PATH, "train.csv"]), 'r'))
print(train_csv_file)

train_labels = []
image_paths = []
for line in train_csv_file:
    # print(line[0])
    image_path = path.sep.join([config.TEST_DATASET_PATH, line[0]])
    print(image_path)
    train_labels.append(line)

print(train_labels[1])



