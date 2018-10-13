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














