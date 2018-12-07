import numpy as np
import tensorflow as tf


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

a = np.ones([1,10],np.float32)
a[0,5] = 1
b = tf.reduce_mean(a)


print(sess.run(b))
