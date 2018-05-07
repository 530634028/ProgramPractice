#
#  For data loading.
#  date : 2018-5-7
#  a    : zhonghy
#
#

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import os

#1.preload
##x1 = tf.constant([2, 3, 4])
##x2 = tf.constant([4, 0, 1])
##y = tf.add(x1, x2)
##print(y.get_shape())
##init = tf.global_variables_initializer()
##with tf.Session() as sess:
##    sess.run(init)
##    print(sess.run(y))
##    exit()


#2.feeding
##a1 = tf.placeholder(tf.int16)
##a2 = tf.placeholder(tf.int16)
##b = tf.add(a1, a2)
##
##li1 = [2, 3, 4]
##li2 = [4, 0, 1]
##
##with tf.Session() as sess:
##    print(sess.run(b, feed_dict={a1:li1, a2:li2}))


#3.reading from file
def main(unused_argv):
    #obtain data
##    "F:\\data\\MNIST_data"
    data_sets = input_data.read_data_sets(FLAGS.directory,
                                     dtype=tf.uint8, #atention code is uint8
                                     reshape=False,
                                     validation_size=FLAGS.validation_size)
    #convert data to tf.train.Example type, then be written to TFRecords files
    convert_to(data_sets.train, 'train')
    convert_to(data_sets.validation, 'validation')
    convert_to(data_sets.test, 'test')

def convert_to(data_set, name):
    images = data_set.images
    labels = data_set.labels
    num_examples = data_set.num_examples #55000 data for training,
                                         #5000 for verification, 10000 for testing
    if images.shape[0] != num_examples:
        raise ValueError('Images size %d does not match label size %d.' %
                         (images.shape[0], num_examples))
    rows = images.shape[1] #28
    cols = images.shape[2] #28
    depth = images.shapw[3] #1, gray image, so it is single channel

    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(num_examples):
        image_raw = images[index].tostring()

        #write to protocol buffer, height,width,depth,label
        #are coded to int64, image_raw is coded to binary
        example = tf.train.Example(features=tf.train.Features(feature={
            'height':_int64_feature(rows),
            'width':_int64_feature(cols),
            'depth':_int64_feature(depth),
            'label':_int64_feature(int(labels[intdex])),
            'image_raw':_bytes_feature(image_raw)}))
        writer.write(example.SerializeToString()) #serilize
    writer.close()

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

main(1)






