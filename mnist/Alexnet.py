from datetime import datetime
import math
import time
import tensorflow as tf
import numpy as np

batch_size = 32
num_batchs = 100
class_num = 10

keep_prob = tf.placeholder(tf.float32, None,name='keep_prob')
x = tf.placeholder(tf.float32, [None, 224, 224, 3],name='x')
y_ = tf.placeholder(tf.float32, [None, class_num],name='y_')

# 输出网络结构尺寸
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def inference(image):
    parameters = []

    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64],
                                                 dtype=tf.float32, stddev=0.1, name='weight'))
        conv = tf.nn.conv2d(image, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32,shape =  [64]),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]

    lrn1 = tf.nn.lrn(conv1, 4, 1.0, 0.001 / 9, 0.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool1')
    print_activations(pool1)

    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192],
                                                 dtype=tf.float32, stddev=0.1, name='weight'))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32,shape =  [192]),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        print_activations(conv2)
        parameters += [kernel, biases]

    lrn2 = tf.nn.lrn(conv2, 4, 1.0, 0.001 / 9, 0.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool2')
    print_activations(pool2)

    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                 dtype=tf.float32, stddev=0.1, name='weight'))
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32,shape =  [384]),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        print_activations(conv3)
        parameters += [kernel, biases]

    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 dtype=tf.float32, stddev=0.1, name='weight'))
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32,shape =  [256]),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        print_activations(conv4)
        parameters += [kernel, biases]

    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                 dtype=tf.float32, stddev=0.1, name='weight'))
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32,shape =  [256]),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        print_activations(conv5)
        parameters += [kernel, biases]

    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool5')
    print_activations(pool5)

    with tf.name_scope('fc1') as scope:
        flat = tf.reshape(pool5,[-1,6*6*256])
        weights = tf.Variable(tf.truncated_normal([6*6*256,4096]),name='weight')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[4096]),
                             trainable=True, name='biases')
        fc1 = tf.nn.dropout(tf.nn.relu(tf.nn.xw_plus_b(flat,weights,biases))
                            ,keep_prob=keep_prob,name=scope)
        print_activations(fc1)
        parameters += [weights, biases]

    with tf.name_scope('fc2') as scope:
        weights = tf.Variable(tf.truncated_normal([4096,4096]),name='weight')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[4096]),
                             trainable=True, name='biases')
        fc2 = tf.nn.dropout(tf.nn.relu(tf.nn.xw_plus_b(fc1,weights,biases))
                            ,keep_prob=keep_prob,name=scope)
        print_activations(fc2)
        parameters += [weights, biases]

    with tf.name_scope('fc3') as scope:
        weights = tf.Variable(tf.truncated_normal([4096,1000]),name='weight')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[1000]),
                             trainable=True, name='biases')
        fc3 = tf.nn.dropout(tf.nn.relu(tf.nn.xw_plus_b(fc2,weights,biases))
                            ,keep_prob=keep_prob,name=scope)
        print_activations(fc3)
        parameters += [weights, biases]

    return fc3, parameters


def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    toatl_duration_squared = 0.0

    for i in range(num_batchs + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target,feed_dict={keep_prob:0.5})
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                      (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            toatl_duration_squared += duration * duration
    mn = total_duration / num_batchs
    vr = toatl_duration_squared / num_batchs - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, num_batchs, mn, sd))

def run_benchmark():
    # with tf.Graph().as_default():
    image_size = 224
    images = tf.Variable(tf.random_normal([batch_size,
                                           image_size,
                                           image_size,3],
                                          dtype=tf.float32,
                                          stddev=0.1))
    fc3,parameters = inference(images)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    time_tensorflow_run(sess,fc3,'Forward')

    objective = tf.nn.l2_loss(fc3)
    grad = tf.gradients(objective,parameters)
    time_tensorflow_run(sess,grad,'Forward-backward')

run_benchmark()