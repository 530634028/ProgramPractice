
##print("Python")
##print("\tPython")

#print("Languages:\n\tPython\n\tC\n\tJavaScript")

##f = 'python '
##print(f.rstrip())
##print(f)
##f = f.rstrip()
##print(f)

##f = ' python '
##print(f.rstrip())  # end
##print(f.lstrip())  # start
##print(f.strip())   # both end and start

##m = "one of Pthon's is"
##print(m)
##m = 'one of Pthon's is'
##print(m)
##
##print(2 + 3)

##bicycles = ['trek', 'cannondale', 'redline', 'specialized']
##print(bicycles)
##print(bicycles[0])
##print(bicycles[0].title())
##print(bicycles[1])
##print(bicycles[3])
##print(bicycles[-1])
##print(bicycles[-2])

##motorcycles = ['honda', 'yamaha', 'suzuki']
##print(motorcycles)
##motorcycles.append('ducati')
##print(motorcycles)
##motorcycles.insert(0, 'ducati')
##print(motorcycles)
##del motorcycles[0]
##print(motorcycles)
##popped_motorcycle = motorcycles.pop()
##print(motorcycles)
##print(popped_motorcycle)
##motorcycles.remove('honda')
##print(motorcycles)

import tensorflow as tf
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))




























