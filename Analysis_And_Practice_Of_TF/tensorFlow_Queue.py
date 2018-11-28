#
#
#   Test Queue
#   date: 2018-5-3
#   a   : zhonghy


import tensorflow as tf

#create FIFO queue
q = tf.FIFOQueue(3, "float")
init = q.enqueue_many(([0.1, 0.2, 0.3],))

#define dequeue, +1, enqueue
x = q.dequeue()
y = x + 1
q_inc = q.enqueue([y])

with tf.Session() as sess:
    sess.run(init)
    quelen = sess.run(q.size())
    for i in range(2):
        sess.run(q_inc)  #operate 2 times, put values in queue

    quelen = sess.run(q.size())
    for i in range(quelen):
        print(sess.run(q.dequeue())) #queue out the values

#create RandomShuffleQueue
print("RandomShuffleQueue: ")
q = tf.RandomShuffleQueue(capacity=10, min_after_dequeue=2, dtypes="float")
sess = tf.Session();

for i in range(0, 10):  #pip 10 times
    sess.run(q.enqueue(i))

for i in range(0, 8): #pop 8 times
    print(sess.run(q.dequeue()))

###discard the interdict
##run_options = tf.RunOptions(timeout_in_ms = 10000)
##try:
##    sess.run(q.dequeue(), options=run_options)
##except tf.errors.DeadlineExceededError:
##    print('out of range')







