#
# Use softmax regression to classify mnist dataset
# date: 2018-5-15
# a   : zhonghy
#
#

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

#load data
mnist = input_data.read_data_sets('F:\data\MNIST_data', one_hot=True)

#define regression model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b #predict value

#define loss function and optimizer
y_ = tf.placeholder(tf.float32, [None, 10]) #input placeholder of real value
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y, labels = y_))

#util SGD optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#use interactive session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#Train, loop 1000 and batch size is 100
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

#estimate model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) #calculate predict and real value
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))#transform bool to float

#calculate accuracy in test dataset
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))

