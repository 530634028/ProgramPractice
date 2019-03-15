"""
 read annotates for cars classification
 a : zhonghy
 date: 2018-3-14

"""

import scipy.io as scio
import numpy as np
# import pandas as pd
from config import car_config as config
import os

def read_annotates():

    #Method 1
    data_path="car_devkit/devkit/cars_annos.mat"
    data = scio.loadmat(data_path) # , struct_as_record=True)
    data_train_class_names = data.get('class_names')
    data_train_annotates = data.get('annotations')

    # print(data_train_class_names)
    data_train_label = data_train_annotates['class']
    data_train_fname = data_train_annotates['relative_im_path']  #data_train_annotates also has # # ('bbox_x1', 'O'), ('bbox_y1', 'O'),
                                                                 # ('bbox_x2', 'O'),('bbox_y2', 'O'), ('class', 'O'), ('test', 'O'))
    # data_train_label = data_train_label.tolist()
    # data_train_fname = data_train_fname.tolist()

    # # for test
    # numOfLabes = np.size(data_train_label)
    # print(numOfLabes - 1)
    #
    # firstClassnames = data_train_class_names[0][1]
    # print(firstClassnames[0][:])
    #
    # firstArray = data_train_label[0][0]
    # print(firstArray[0][0])
    #
    # print(type(data_train_label)) # [0][0])
    print(data_train_class_names.shape)

    trainPaths = []
    trainLabels = []
    for index in range(0, np.size(data_train_fname)):  # np.size(data_train_fname)
        fileNameTmp = data_train_fname[0][index]
        fileName = fileNameTmp[0][:]
        fileName = fileName[fileName.rfind("/") + 1:]
        filePath = os.sep.join([config.IMAGES_PATH, fileName])
        # print(filePath)
        # print(fileName)
        trainPaths.append(filePath)

        labelIndexTmp = data_train_label[0][index]
        labelIndex = labelIndexTmp[0][0]
        # print(labelIndex)
        classnameTmp = data_train_class_names[0][labelIndex - 1] # in the class fields of data_train_annotates, it is start at 1,
                                                                 # so subract 1  attention
        classname = classnameTmp[0][:]
        trainLabels.append(classname)
        # print(classname)

    ImagePathAndClass = [trainPaths, trainLabels]   # error, code in for loop
    # print(trainPaths)
    return ImagePathAndClass

    # method 2
    # from pandas import Series,DataFrame
    # import pandas as pd
    # import numpy as np
    # import h5py
    # datapath = data_path
    # file = h5py.File(datapath,'r')
    # def Print(name):print(name)
    # data = file['CH01'][:]
    # dfdata = pd.DataFrame(data)
    # datapath1 = 'data3.txt'
    # dfdata.to_csv(datapath1)