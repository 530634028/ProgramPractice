#
# Training AlexNet on dogs vs cats dataset
# a : zhonghy
# date: 2018-9-3
#

# import the necessary packages
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

import sys;
sys.path.append("F:\ProgramPractice\DLFCV")

from config import dogs_vs_cats_config as config
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.patchpreprocessor import PatchPreprocessor
from pyimagesearch.preprocessing.meanpreprocessor import MeanPreprocessor
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from pyimagesearch.io.hdf5datasetgenerator import HDF5DatasetGenerator
from pyimagesearch.nn.conv.alexnet import AlexNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import json
import os

bSize = 8  # 128

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2,
                         shear_range=0.15, horizontal_flip=True,
                         fill_mode="nearest")

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, bSize, aug=aug,
                                preprocessors=[pp, mp, iap],
                                classes=2)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, bSize,
                              preprocessors=[sp, mp, iap],
                              classes=2)

# initialize the optimizer
opt = Adam(lr=1e-3)
model = AlexNet.build(width=227, height=227, depth=3,
                      classes=2, reg=0.0002)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# construct the set of callbacks
path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(
    os.getpid())])
callbacks = [TrainingMonitor(path)]

# train the network
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // bSize,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // bSize,
    epochs=75,
    max_queue_size=bSize * 2,
    callbacks=callbacks, verbose=1
)

# save the model to file
print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)

# clost the HDF5 datasets
trainGen.close()
valGen.close()

