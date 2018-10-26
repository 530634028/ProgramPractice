#
# we’ll use to generate the .lst files for the training, testing,
# and validation splits, respectively
# a : zhonghy
# date: 2018-10-11
#

# import the necessary packages
import numpy as np
import os

class ImageNetHelper:
    def __init__(self, config):
        # store the configuration object
        self.config = config

        # build the label mappings and validation blacklist
        self.labelMappings = self.buildClassLabels()
        self.valBlacklist = self.buildBlacklist()

    def buildClassLables(self):
        # load the contents of the filet that maps the WordNet IDs
        # to integers, then initialize the label mappings dictionary
        rows = open(self.config.WORD_IDS).read().strip().split("\n")
        labelMappings = {}

        # loop over the labels
        for row in rows:
            # split the row into the WordNet ID, label integer, and
            # human readable label
            (wordID, label, hrLabel) = row.split(" ")

            # update the label mappings dictionary using the word ID
            # as the key and the label as the value, subtracting a"
            # from the label since MATLAB is one-indexed while Python
            # is zero-indexed
            labelMappings[wordID] = int(label) - 1

        # return the label mappings dictionary
        return labelMappings

    def buildBlacklist(self):
        # load the list of blacklisted image IDs and convert them to
        # a set
        rows = open(self.config.VAL_BLACKLIST).read()
        rows = set(rows.strip().split("\n"))

        # return the blacklisted image IDs
        return rows

    def buildTrainingSet(self):
        # load the contents of the training input file that lists
        # the partial image ID and image number, then initialize
        # the list of image paths and class labels
        rows = open(self.config.TRAIN_LIST).read().strip()
        rows = rows.split("\n")
        paths = []
        labels = []

        # loop over the rows in the input training file
        for row in rows:
            # break the row into the partial path and image
            # number (the image number is sequential and is
            # essentially useless to us )
            (partialPath, imageNum) = row.strip().split(" ")

            # construct the full path to the training image, then
            # grab the word ID from the path and use it to determine
            # the integer classs label
            path = os.path.sep.join([self.config.IMAGES_PATH,
                                     "train", "{}.JEPG".format(partialPath)])
            wordID = partialPath.split("/")[0]
            label = self.labelMappings[wordID]

            # update the respective paths and label lists
            paths.append(path)
            labels.append(label)

        # return a tuple of image paths and associated integer class
        # labels
        return (np.array(paths), np.array(labels))

    def buildValidationSet(self):
        # initialize the list of image paths and class labels
        paths = []
        labels = []






















