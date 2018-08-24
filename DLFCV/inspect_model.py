#
# Let’s go ahead and take a look at the layer names and indexes in VGG16
# a : zhonghy
# date: 2018-8-22
#

# import the necessary packages
from keras.applications import VGG16
import argparse

# construct the argument and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--include-top",type=int, default=1,
                help="whether or not to include top of CNN")
args = vars(ap.parse_args())

# loading the VGG16 network
print("[INFO] loading network...")
model = VGG16(weights="imagenet",
              include_top=args["include_top"] > 0)
print("[INFO] showing layers...")

# loop over the layers in the network and display them to the
# console
for (i, layer) in enumerate(model.layers):
    print("[INFO] {}\t{}".format(i, layer.__class__.__name__))





