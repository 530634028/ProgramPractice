"""
# Save model every NUM epochs
# a : zhonghy
# date: 2018-9-6

"""

# import necessary packages
from keras.callbacks import ModelCheckpoint
import argparse
from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import keras as K
import os

class EpochCheckpoint(K.callbacks.Callback):
    def __init__(self, startPath, every=1, startAt=0):
        super(EpochCheckpoint, self).__init__()
        self.startPath = startPath
        self.every = every
        self.startAt = startAt

    def on_epoch_end(self, epoch, log={}):
        # construct the callback to save only the *best* model to disk
        # based on the validation loss

        #if epoch % 1 == 0:
        fname = os.path.sep.join([self.startPath,
                                      "weights-{epoch:03d}.hdf5"])
        ModelCheckpoint(fname, monitor="val_acc", mode="max")
            #checkpoint =
            #return checkpoint
            #callbacks = [checkpoint]
