"""
 build dataset for cars classification
 a : zhonghy
 date: 2018-1-8

"""

# import the necessary packages
import sys
# sys.path.append("F:\ProgramPractice\DLFCV")
sys.path.append("E:\ZWDX_Learn\ProgramPractice\DLFCV")

from config import car_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from car_classification.read_annotates import read_annotates
import progressbar
import pickle

# read the contents of the labels file, then initialize the list of
# image path and labels
print("[INFO] loading image paths and labels...")
# rows = open(config.LABELS_PATH).read()
# rows = rows.strip().split("\n")[1:]
# trainPaths = []
# trainLabels = []
#
# # loop over the rows
# for row in rows:
#     # unpack the row, then update the image paths and labels list
#     # (filename, make) = row.split(",")[:2]
#     (filename, make, model) = row.split(",")[:3]
#     filename = filename[filename.rfind("/") + 1:]
#     trainPaths.append(os.sep.join([config.IMAGES_PATH, filename]))
#     trainLabels.append("{}:{}".format(make, model))


# test zhong
(trainPaths, trainLabels)= read_annotates()
# print(trainPaths.shape)

# now that we have the total number of images in the dataset that
# can be used for training, compute the number of images that
# should be used for validation and testing
numVal = int(len(trainPaths) * config.NUM_VAL_IMAGES)
numTest = int(len(trainPaths) * config.NUM_TEST_IMAGES)

# our class labels are represent as strings so we need to encode
# them
print("[INFO] encoding labels...")
le = LabelEncoder().fit(trainLabels)
trainLabels = le.transform(trainLabels)

# test transform and inverse_transform function
# print(trainLabels)

# perform sampling from the training set to construct a validation
# set
print("[INFO] constructing validation data...")
split = train_test_split(trainPaths, trainLabels, test_size=numVal,
                         stratify=trainLabels)
(trainPaths, valPaths, trainLabels, valLabels) = split

# perform stratified sampling from the training set to construct a
# testing set
print("[INFO] constructing testing data...")
split = train_test_split(trainPaths, trainLabels, test_size=numTest,
                         stratify=trainLabels)
(trainPaths, testPaths, trainLabels, testLabels) = split

# construct a list pairing the training, validation, and testing
# image paths along with their corresponding labels and output loist
# files
datasets = [
    ("train", trainPaths, trainLabels, config.TRAIN_MX_LIST),
    ("val", valPaths, valLabels, config.VAL_MX_LIST),
    ("test", testPaths, testLabels, config.TEST_MX_LIST)
]

# loop over the dataset tuples
for (dType, paths, labels, outputPath) in datasets:
    # open the output file for writing
    print("[INFO] building {}...".format(outputPath))
    f = open(outputPath, "w")

    # initialize the progress bar
    widgets = ["Building List: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths),
                                   widgets=widgets).start()

    # loop over each of the individual images + labels
    for (i, (path, label)) in enumerate(zip(paths, labels)):
        # write the image index, label, and output path to file
        row = "\t".join([str(i), str(label), path])
        f.write("{}\n".format(row))
        pbar.update(i)

    # close the output file
    pbar.finish()
    f.close()

# write the label encoder to file
print("[INFO] serializing label encoder...")
f = open(config.LABEL_ENCODER_PATH, "wb")
f.write(pickle.dumps(le))
f.close()


# then use mx command to generate .rec files