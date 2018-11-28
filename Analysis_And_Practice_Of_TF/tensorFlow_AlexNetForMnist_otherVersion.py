##
##
##
##
### 输入数据
##from tensorflow.examples.tutorials.mnist import input_data
##import tensorflow as tf
##import random
##
##mnist = input_data.read_data_sets("F:/data/MNIST_data", one_hot=True)
##
##import tensorflow as tf
##
### 定义网络超参数
##learning_rate = 0.001
##training_iters = 200000
##batch_size = 64
##display_step = 20
##
### 定义网络参数
##n_input = 784 # 输入的维度
##n_classes = 10 # 标签的维度
##dropout = 0.8 # Dropout 的概率
##
### 占位符输入
##x = tf.placeholder(tf.float32, [None, n_input])
##y = tf.placeholder(tf.float32, [None, n_classes])
##keep_prob = tf.placeholder(tf.float32)
##
### 卷积操作
##def conv2d(name, l_input, w, b):
##    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)
##
### 最大下采样操作
##def max_pool(name, l_input, k):
##    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)
##
### 归一化操作
##def norm(name, l_input, lsize=4):
##    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
##
### 定义整个网络 
##def alex_net(_X, _weights, _biases, _dropout):
##    # 向量转为矩阵
##    _X = tf.reshape(_X, shape=[-1, 28, 28, 1])
##
##    # 卷积层
##    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
##    # 下采样层
##    pool1 = max_pool('pool1', conv1, k=2)
##    # 归一化层
##    norm1 = norm('norm1', pool1, lsize=4)
##    # Dropout
##    norm1 = tf.nn.dropout(norm1, _dropout)
##
##    # 卷积
##    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
##    # 下采样
##    pool2 = max_pool('pool2', conv2, k=2)
##    # 归一化
##    norm2 = norm('norm2', pool2, lsize=4)
##    # Dropout
##    norm2 = tf.nn.dropout(norm2, _dropout)
##
##    # 卷积
##    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
##    # 下采样
##    pool3 = max_pool('pool3', conv3, k=2)
##    # 归一化
##    norm3 = norm('norm3', pool3, lsize=4)
##    # Dropout
##    norm3 = tf.nn.dropout(norm3, _dropout)
##
##    # 全连接层，先把特征图转为向量
##    dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]]) 
##    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1') 
##    # 全连接层
##    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
##
##    # 网络输出层
##    out = tf.matmul(dense2, _weights['out']) + _biases['out']
##    return out
##
### 存储所有的网络参数
##weights = {
##    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
##    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
##    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
##    'wd1': tf.Variable(tf.random_normal([4*4*256, 1024])),
##    'wd2': tf.Variable(tf.random_normal([1024, 1024])),
##    'out': tf.Variable(tf.random_normal([1024, 10]))
##}
##biases = {
##    'bc1': tf.Variable(tf.random_normal([64])),
##    'bc2': tf.Variable(tf.random_normal([128])),
##    'bc3': tf.Variable(tf.random_normal([256])),
##    'bd1': tf.Variable(tf.random_normal([1024])),
##    'bd2': tf.Variable(tf.random_normal([1024])),
##    'out': tf.Variable(tf.random_normal([n_classes]))
##}
##
### 构建模型
##pred = alex_net(x, weights, biases, keep_prob)
##
### 定义损失函数和学习步骤
##cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
##optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
##
### 测试网络
##correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
##accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
##
### 初始化所有的共享变量
##init = tf.global_variables_initializer()
##
### 开启一个训练
##with tf.Session() as sess:
##    sess.run(init)
##    step = 1
##    # Keep training until reach max iterations
##    while step * batch_size < training_iters:
##        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
##        # 获取批数据
##        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
##        if step % display_step == 0:
##            # 计算精度
##            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
##            # 计算损失值
##            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
##            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
##        step += 1
##    print("Optimization Finished!")
##    # 计算测试精度
##    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))



# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Timing benchmark for AlexNet inference.
To run, use:
  bazel run -c opt --config=cuda \
      models/tutorials/image/alexnet:alexnet_benchmark
Across 100 steps on batch size = 128.
Forward pass:
Run on Tesla K40c: 145 +/- 1.5 ms / batch
Run on Titan X:     70 +/- 0.1 ms / batch
Forward-backward pass:
Run on Tesla K40c: 480 +/- 48 ms / batch
Run on Titan X:    244 +/- 30 ms / batch
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import math
import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

FLAGS = None


def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())


def inference(images):
  """Build the AlexNet model.
  Args:
    images: Images Tensor
  Returns:
    pool5: the last Tensor in the convolutional component of AlexNet.
    parameters: a list of Tensors corresponding to the weights and biases of the
        AlexNet model.
  """
  parameters = []
  # conv1
  with tf.name_scope('conv1') as scope:
    kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope)
    print_activations(conv1)
    parameters += [kernel, biases]

  # lrn1
  with tf.name_scope('lrn1') as scope:
    lrn1 = tf.nn.local_response_normalization(conv1,
                                              alpha=1e-4,
                                              beta=0.75,
                                              depth_radius=2,
                                              bias=2.0)

  # pool1
  pool1 = tf.nn.max_pool(lrn1,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool1')
  print_activations(pool1)

  # conv2
  with tf.name_scope('conv2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
  print_activations(conv2)

  # lrn2
  with tf.name_scope('lrn2') as scope:
    lrn2 = tf.nn.local_response_normalization(conv2,
                                              alpha=1e-4,
                                              beta=0.75,
                                              depth_radius=2,
                                              bias=2.0)

  # pool2
  pool2 = tf.nn.max_pool(lrn2,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool2')
  print_activations(pool2)

  # conv3
  with tf.name_scope('conv3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv3)

  # conv4
  with tf.name_scope('conv4') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv4)

  # conv5
  with tf.name_scope('conv5') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv5)

  # pool5
  pool5 = tf.nn.max_pool(conv5,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool5')
  print_activations(pool5)

  return pool5, parameters


def time_tensorflow_run(session, target, info_string):
  """Run the computation to obtain the target tensor and print timing stats.
  Args:
    session: the TensorFlow session to run the computation under.
    target: the target Tensor that is passed to the session's run() function.
    info_string: a string summarizing this run, to be printed with the stats.
  Returns:
    None
  """
  num_steps_burn_in = 10
  total_duration = 0.0
  total_duration_squared = 0.0
  for i in xrange(FLAGS.num_batches + num_steps_burn_in):
    start_time = time.time()
    _ = session.run(target)
    duration = time.time() - start_time
    if i >= num_steps_burn_in:
      if not i % 10:
        print ('%s: step %d, duration = %.3f' %
               (datetime.now(), i - num_steps_burn_in, duration))
      total_duration += duration
      total_duration_squared += duration * duration
  mn = total_duration / FLAGS.num_batches
  vr = total_duration_squared / FLAGS.num_batches - mn * mn
  sd = math.sqrt(vr)
  print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
         (datetime.now(), info_string, FLAGS.num_batches, mn, sd))



def run_benchmark():
  """Run the benchmark on AlexNet."""
  with tf.Graph().as_default():
    # Generate some dummy images.
    image_size = 224
    # Note that our padding definition is slightly different the cuda-convnet.
    # In order to force the model to start with the same activations sizes,
    # we add 3 to the image_size and employ VALID padding above.
    images = tf.Variable(tf.random_normal([FLAGS.batch_size,
                                           image_size,
                                           image_size, 3],
                                          dtype=tf.float32,
                                          stddev=1e-1))

    # Build a Graph that computes the logits predictions from the
    # inference model.
    pool5, parameters = inference(images)

    # Build an initialization operation.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=config)
    sess.run(init)

    # Run the forward benchmark.
    time_tensorflow_run(sess, pool5, "Forward")

    # Add a simple objective so we can calculate the backward pass.
    objective = tf.nn.l2_loss(pool5)
    # Compute the gradient with respect to all the parameters.
    grad = tf.gradients(objective, parameters)
    # Run the backward benchmark.
    time_tensorflow_run(sess, grad, "Forward-backward")


def main(_):
  run_benchmark()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--batch_size',
      type=int,
      default=128,
      help='Batch size.'
  )
  parser.add_argument(
      '--num_batches',
      type=int,
      default=100,
      help='Number of batches to run.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
