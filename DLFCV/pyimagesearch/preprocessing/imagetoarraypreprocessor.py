#
# Transform input image to NumPy array for training and testing
# a: zhonghy
# date: 2018-7-18
#
#

# import the necessary packages
from keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # store the image data format
        self.dataFormat = dataFormat

    def preprocess(self, image):
        # apply the Keras utility function that correctly rearranges
        # the dimensions of image
        return img_to_array(image, data_format=self.dataFormat)
    
        
