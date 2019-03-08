"""
 training for age and gender classification
 a : zhonghy
 date: 2019-1-15

"""

# import the necessary packages
import sys
sys.path.append("F:\ProgramPractice\DLFCV")

from config import age_gender_config as config
from pyimagesearch.nn.mxconv.mxagegender import MxAgeGenderNet
from pyimagesearch.utils.agegemderhelper import AgeGenderHelper #???
from pyimagesearch.mxcallbacks.mxmetrics import one_off_callback
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
import mxnet as mx
import argparse
import logging
import pickle
import json
import os

from os import path

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
                help="path to output checkpoint directory")
ap.add_argument("-p", "--prefix", required=True,
                help="name of model prefix")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
                help="epoch to restart training at")
args = vars(ap.parse_args())

# set the logging level and output file
logging.basicConfig(level=logging.DEBUG,
                    filename="training_{}.log".format(args["start_epoch"]),
                    filemode="w")

# load the RGB means for the training set, then determin the batch
# size
means = json.loads(open(config.DATASET_MEAN).read())
batchSize = config.BATCH_SIZE * config.NUM_DEVICES

# construct the training image iterator
trainIter = mx.io.ImageRecordIter(
    path_imgrec=config.TRAIN_MX_REC,
    data_shape=(3, 227, 227),
    batch_size=batchSize,
    rand_crop=True,
    rand_mirror=True,
    rotate=7,
    mean_r=means["R"],
    mean_g=means["G"],
    mean_b=means["B"],
    preprocess_threads=config.NUM_DEVICES * 2
)

# construct the validation image iterator
valIter = mx.io.ImageRecordIter(
    path_imgrec=config.VAL_MX_REC,
    data_shape=(3, 227, 227),
    batch_size=batchSize,
    mean_r=means["R"],
    mean_g=means["G"],
    mean_b=means["B"]
)

# initialize the optimmizer
opt = mx.optimizer.Adam(learning_rate=1e-5) #, momentum=0.9, wd=0.0005,  # fine tuning value le-3 le-5 le-6 SGD
                       # rescale_grad=1.0 / batchSize)

# SGD(learning_rate=1e-4, momentum=0.9, wd=0.0005,     # fine tuning value le-3 le-5 le-6 SGD
#                        rescale_grad=1.0 / batchSize)

# construct the checkpoints path, initialize the model argument and
# auxiliary parameters
checkpointsPath = os.path.sep.join([args["checkpoints"],
                                    args["prefix"]])
#print(checkpointsPath)

argParams = None
auxParams = None

# if there is no specific model starting epoch supplied, then
# initialize the network
if args["start_epoch"] <= 0:
    # build the AlexNet architecture
    print("[INFO] building network...")
    model = MxAgeGenderNet.build(config.NUM_CLASSES)
# otherwise, a specific checkpoint was supplied
else:
    # load the checkpoint from disk
    print("[INFO] loading epoch {}...".format(args["start_epoch"]))
    (model, argParams, auxParams) = mx.model.load_checkpoint(
        checkpointsPath, args["start_epoch"]
    )
    #print(len(model))

# compile the model
model = mx.model.FeedForward(
    ctx=[mx.gpu(0)], # mx.gpu(3)], # by your need
    symbol=model,
    initializer=mx.initializer.Xavier(),
    arg_params=argParams,
    aux_params=auxParams,
    optimizer=opt,
    num_epoch=80,   # change as you need, original value is 110
    begin_epoch=args["start_epoch"]
)

# intialize the callbacks and evaluation metrics
batchEndCBs = [mx.callback.Speedometer(batchSize, 10)]
epochEndCBs = [mx.callback.do_checkpoint(checkpointsPath)]
metrics = [mx.metric.Accuracy(), mx.metric.CrossEntropy()] #, mx.metric.Loss()]

# check to see if the one-off accuracy callback should be used
if config.DATASET_MEAN == "age":
    # load the label encoder, then build the one-off mappings for
    # computing accuracy
    le = pickle.loads(open(config.LABEL_ENCODER_PATH, "rb").read())
    agh = AgeGenderHelper(config)
    oneOff = agh.buildOneOffMappings(le)
    epochEndCBs.append(one_off_callback(trainIter, valIter,
                                        oneOff, mx.gpu(0)))

figureOfTraining = path.sep.join([config.OUTPU_BASE, "age_training.png"])
trainingMonitor = TrainingMonitor(figureOfTraining)
                                   # startAt=args["start_epoch"])

# train the network
print("[INFO] training network...")
model.fit(
    X=trainIter,
    eval_data=valIter,
    eval_metric=metrics,
    batch_end_callback=batchEndCBs,
    epoch_end_callback=epochEndCBs #, trainingMonitor] # TypeError: 'list' object is not callable
)

# python train.py --checkpoints checkpoints/age --prefix agenet