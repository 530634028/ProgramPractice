"""
 build lisa dataset for ssds and rcnn
 -- traffic signs detector
 a : zhonghy
 date: 2019-1-23

"""

# import the necessary packages
import sys
# sys.path.append("F:\ProgramPractice\DLFCV")
sys.path.append("E:\ZWDX_Learn\ProgramPractice\DLFCV")

from config import lisa_config as config
from pyimagesearch.utils.tfannotation import TFAnnotation
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
import os

def main(_):
    # open the classes output file
    f = open(config.CLASSES_FILE, "w")

    # loop over the classes
    for (k, v) in config.CLASSES.items():
        # construct the class information and write to file
        item = ("item {\n"
                "\tid: " + str(v) + "\n"
                "\tname: '" + k + "'\n"
                "}\n")
        f.write(item)

    # close the output classes file
    f.close()

    # initialize a data dictionary, then load the annotations file
    D = {}
    rows = open(config.ANNOT_PATH).read().strip().split("\n")

    # loop over the individual rows, skipping the header
    for row in rows[1:]:
        # break the row into components
        row = row.split(",")[0].split(";")
        (imagePath, label, startX, startY, endX, endY, _) = row
        (startX, startY) = (float(startX), float(startY))
        (endX, endY) = (float(endX), float(endY))

        # if we are not interested in the label, ignore it
        if label not in config.CLASSES:
            continue

        # build the path to then input image, then grab any other
        # bounding boxes + labels associated with the image
        # path, labels, and bounding box lists, respectively
        p = os.path.sep.join([config.BASE_PATH, imagePath])
        b = D.get(p, [])

        # build a tuple consisting of the label and bounding box
        # then update teh list and store it in dictionary
        b.append((label, (startX, startY, endX, endY)))
        D[p] = b

    # create training and testing splits from our data dictionary
    (trainKeys, testKeys) = train_test_split(list(D.keys()),
                                             test_size=config.TEST_SIZE,
                                             random_state=42)
    # initialize the data split files
    datasets = [
        ("train", trainKeys, config.TRAIN_RECORD),
        ("test", testKeys, config.TEST_RECORD)
    ]

    # loop over the datasets
    for (dType, keys, outputPath) in datasets:
        # initialize the TensorFlow writer and initialize the total
        # number of examples written to file
        print("[INFO] processing '{}'...".format(dType))
        writer = tf.python_io.TFRecordWriter(outputPath)
        total = 0

        # loop over all the keys in then current set
        for k in keys:
            # load the input image from disk as a TensorFlow object
            encoded = tf.gfile.GFile(k, "rb").read()
            encoded = bytes(encoded)

            # load the image from disk again, this time as a PIL
            # object
            pilImage = Image.open(k)
            (w, h) = pilImage.size[:2]

            # parse the filename and encoding from the input path
            filename = k.split(os.path.sep)[-1]
            encoding = filename[filename.rfind(".") + 1:]

            # initialize the annotation object used to store
            # information regarding the bounding box + labels
            tfAnnot = TFAnnotation()
            tfAnnot.image = encoded
            tfAnnot.encoding = encoding
            tfAnnot.filename = filename
            tfAnnot.width = w
            tfAnnot.height = h

            # loop over the bounding boxes + labels associated with
            # the image
            for (label, (startX, startY, endX, endY)) in D[k]:
                # TensorFlow assumes all bounding boxes are in the
                # range [0, 1] so we need to scale them
                xMin = startX / w
                xMax = endX / w
                yMin = startY / h
                yMax = endY / h

                """
                # double-check for the dataset, read image from
                # disk and show it in screen, code block
                image = cv2.imread(k)
                startX = int(xMin * w)
                startY = int(yMin * h)
                endX = int(xMax * w)
                endY = int(yMax * h)
                
                # draw the bounding box on the image
                cv2.rectangle(image, (startX, startY), (endX, endY),
                (0, 255, 0), 2)
                
                # show the output image
                cv2.imshow("image", image)
                cv2.waitKey(0)
                
                """

                # update the bounding boxe + labels list
                tfAnnot.xMins.append(xMin)
                tfAnnot.xMaxs.append(xMax)
                tfAnnot.yMins.append(yMin)
                tfAnnot.yMaxs.append(yMax)
                tfAnnot.textLabels.append(label.encode("utf8"))
                tfAnnot.classes.append(config.CLASSES[label])
                tfAnnot.difficult.append(0)

                # increment the total number of examples
                total += 1

            # encode the data point attributes using the TensorFlow
            # helper functions
            features = tf.train.Features(feature=tfAnnot.build())
            example = tf.train.Example(features=features)

            # add the example to the writer
            writer.write(example.SerializeToString())

        # close the writer and print diagnostic information to the
        # user
        writer.close()
        print("[INFO] {} examples saved for '{}'".format(total,
                                                         dType))

# check to see if the main thread should be started
if __name__ == "__main__":
    tf.app.run()

# time python build_lisa_records.py

