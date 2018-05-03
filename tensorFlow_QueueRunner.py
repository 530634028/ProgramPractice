#
#
#
#
#
#
#

import tensorflow as tf

q = tf.FIFOQueue(1000, "float")
counter = tf.Variable(0.0)   #counter
increment_op = tf.assign_add(counter, tf.constant(1.0))
enqueue_op = q.enqueue(counter) #opetration: add counter to queue


qr = tf.train.QueueRunner(q, enqueue_ops=[increment_op, enqueue_op] * 1)

###main thread
##with tf.Session() as sess:
##    sess.run(tf.global_variables_initializer())
##    enqueue_threads = qr.create_threads(sess, start=True) #lanuch pip
##    #main thread
##    for i in range(10):
##        print(sess.run(q.dequeue()))


#main thread    
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Coordinator
coord = tf.train.Coordinator()

#lanuch pip
enqueue_threads = qr.create_threads(sess, coord = coord, start=True) #lanuch pip


coord.request_stop()
#main thread
for i in range(0, 10):
    try:
        print(sess.run(q.dequeue()))
    except tf.errors.OutOfRangeError:
        break

#info other threads to close
##coord.request_stop()
coord.join(enqueue_threads)




    




