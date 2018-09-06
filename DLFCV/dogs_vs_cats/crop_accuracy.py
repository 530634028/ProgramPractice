#
# To evaluate AlexNet on the testing set using both our standard
# method and over-sampling technique
# a : zhonghy
# date: 2018-9-4
#

# import the necessary packages
from config import dogs_vs_cats_config as config
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.meanpreprocessor import MeanPreprocessor
from pyimagesearch.preprocessing.croppreprocessor import CropPreprocessor
from pyimagesearch.io.hdf5datasetgenerator import HDF5DatasetGenerator
from pyimagesearch.utils.ranked import rank5_accuarcy
from keras.models import load_model
import numpy as np
import progressbar
import json

# load the RGB menas for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
cp = CropPreprocessor(227, 227)
iap = ImageToArrayPreprocessor()

# load the pre-trained network
print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)

# initialize the testing dataset generator, then make predictions on
# the testing data
print("[INFO] predicting on test data (no crops)...")
testGen = HDF5DatasetGenerator(config.TEST_HDF5, 64,
                               preprocessors=[sp, mp, iap],
                               classes=2)
predictions = model.predict_generator(testGen.generator(),
                                      steps=testGen.numImages // 64,
                                      max_queue_size=64 * 2)
# compute the rank-1 and rank-5 accuracies
(rank1, _) = rank5_accuarcy(predictions, testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
testGen.close()

# re-initialize the testing dataset generator,
# this time excluding the 'SimplePreprocessor'
testGen = HDF5DatasetGenerator(config.TEST_HDF5, 64,
                               preprocessors=[mp],
                               classes=2)
predictions =[]

# initialize the progress bar
widgets = ["Evaluating: ", progressbar.Percentage(), " ",
           progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=testGen.numImages // 64,
                               widgets=widgets).start()

# loop over a single pass of the test data
for (i, (images, labels)) in enumerate(testGen.generator(passes=1)):
    # loop over each of the individual images
    for image in images:
        # apply the crop preprocessor to the image to generate 10
        # seperate crops, then convert them from images to arrays
        crops = cp.preprocess(image)
        crops = np.array([iap.preprocess(c) for c in crops],
                         dtype="float32")

        # make predictions on the crops and then average them
        # together to obtain the final prediction
        pred = model.predict(crops)
        predictions.append(pred.mean(axis=0))

    # update the progress bar
    pbar.update(i)

# comput the rank-1 accuracy
pbar.finish()
print("[INFO] predicting on test data (with crops)..")
(rank1, _) = rank5_accuarcy(predictions, testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
testGen.close()



