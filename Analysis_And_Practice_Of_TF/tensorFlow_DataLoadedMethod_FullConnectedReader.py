#
#
#  description : Used to reader data from TFRecord files
#  date        : 2018-5-8
#  a           : zhonghy
#


"""find in tensorflow source code"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
##from tensorflow.contrib.learn.python.learn.datasets import mnist
from tensorflow.examples.tutorials.mnist import mnist

import os
import sys
import argparse
import time

#Basic model parameters as external flags
FLAGS = None

#Constants used for dealing with the files, matches convert_to_records
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'



def read_and_decode(filename_queue): #input filename queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example( #parse example
        serialized_example,
        #must decalare key name in features
        features = {
            'image_raw': tf.FixedLenFeature([], tf.string), #type of images is string
            'label': tf.FixedLenFeature([], tf.int64), #type of labels is int64
            })

    #decode for BytesList
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([mnist.IMAGE_PIXELS])
    #Tensor("input/DecodeRaw:0", shape=(784,), dtype=uint8)

    #shape of image tensor is:Tensor("input/sub:0", shape=(784,), dtype=float32)
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    #convert type of labels uint8 to int32
    #label tensor shpae:Tensor("input/Cast_1:0", shape=(), dtype=int32)
    label = tf.cast(features['label'], tf.int32)

    return image, label


def inputs(train, batch_size, num_epochs):
    if not num_epochs: num_epochs = None
    #obtain file path 'F:\data\MNIST_data'
    filename = os.path.join(FLAGS.train_dir,
                            TRAIN_FILE if train else VALIDATION_FILE)

    with tf.name_scope('input'):
        #return a FIFOQueue QueueRunner
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=num_epochs) #if too big, divide to little batch

    image, label = read_and_decode(filename_queue)

    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size,
        min_after_dequeue=1000)

    return images, sparse_labels

def run_training():
    with tf.Graph().as_default():
        #input images and labes
        images, labels = inputs(train=True, batch_size=FLAGS.batch_size,
                                num_epochs=FLAGS.num_epochs)

        #construct log net
        logits = mnist.inference(images,
                                 FLAGS.hidden1,
                                 FLAGS.hidden2)

        #defin loss function
        loss = mnist.loss(logits, labels)

        #Add to the Graph operations that train the model
        train_op = mnist.training(loss, FLAGS.learning_rate)

        #initilize parameters
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        sess = tf.Session()

        sess.run(init_op)

        #Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while not coord.should_stop(): #in
                start_time = time.time()
                _, loss_value = sess.run([train_op, loss])
                duration = time.time() - start_time
                
             #each 100 times ouput one result
                if step % 100 == 0:
                   print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                step += 1

        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
        finally:
            coord.request_stop() #info other threads to close

        coord.join(threads)
        sess.close()


def main(_):
    run_training()

           
"""Used to run the main function?"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,    #'F:\data\MNIST_data',
        help='Initial learning rate.'
        )

    parser.add_argument(
        '--num_epochs',
        type=int,
        default=2,
        help=' Number of epochs to run trainer'
        )

    parser.add_argument(
        '--hidden1',
        type=int,
        default=128,
        help=' Number of uints in hidden layer 1'
        )
    
    parser.add_argument(
        '--hidden2',
        type=int,
        default=32,
        help=' Number of uints in hidden layer 2'
        )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help=' Batch size'
        )

    parser.add_argument(
        '--train_dir',
        type=str,
        default='F:\data\MNIST_data',
        help=' Directory with the train data'
        )
         
    FLAGS,unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)      



