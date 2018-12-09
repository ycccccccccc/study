import tensorflow as tf
import numpy as np
from datetime import datetime
import math
import time
# import readdata

batch_size = 32
num_batchs = 100
# class_num = 10
# max_step = 10000
#
# model_path = '../../model/Alexnet/'
# model_name = 'model.ckpt'
#
# keep_prob = tf.placeholder(tf.float32, None, name='keep_prob')
# x = tf.placeholder(tf.float32, [None, 224, 224, 3], name='x')
# y_ = tf.placeholder(tf.float32, [None, class_num], name='y_')

slim = tf.contrib.slim

def trunc_normal(stddtv):
    return tf.truncated_normal_initializer(0.0,stddtv)

def inception_v3_arg_scope(weight_decay = 0.00004,
                           stddev = 0.1,
                           batch_norm_var_clooection = 'moving_vars'):
    batch_norm_params = {
        'decay':0.9997,
        'epsilon':0.001,
        'updates_collections':tf.GraphKeys.UPDATE_OPS,
        'variables_collections':{
            'beta':None,
            'gamma':None,
            'moving_mean':[batch_norm_var_clooection],
            'moving_variance':[batch_norm_var_clooection],
        }
    }

    with slim.arg_scope([slim.conv2d,slim.fully_connected],
                        weight_regularizer = slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
            [slim.conv2d],
            weights_initializer = tf.truncated_normal_initializer(stddev=stddev),
            activation_fn = tf.nn.relu,
            normalizer_fn =slim.batch_norm,
            normalizer_params = batch_norm_params) as sc:
            return sc

def inception_v3_base(inputs,scope = None):
    end_points = {}

    with tf.variable_scope(scope,'InceptionV3',[inputs]):
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],
                            stride = 1,padding = 'VALID'):
            net = slim.conv2d(inputs,32,[3,3],strides=2,scope = 'Conv2d_1a_3x3')
            net = slim.conv2d(net,32,[3,3],scope = 'Convd_2a_3x3')
            net = slim.conv2d(net,64,[3,3],padding='SAME',scope = 'Convd_2b_3x3')
            net = slim.max_pool2d(net,[3,3],stride = 2,scope = 'MaxPool_3a_3x3')
            net = slim.conv2d(net, 80, [1,1], scope='Convd_3b_1x1')
            net = slim.conv2d(net, 192, [3, 3], scope='Convd_4a_3x3')
            net = slim.max_pool2d(net,[3, 3], stride=2, scope='MaxPool_5a_3x3')

        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],
                            stride = 1,padding = 'SAME'):
            with










def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + 'w',
                                 shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        return activation


def fc_op(input_op, name, n_out, p,last = False):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + 'w', [n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape = [n_out],
                             dtype=tf.float32), name='b')
        if last:
            activation = tf.nn.xw_plus_b(input_op,kernel,biases,scope)
        else:
            activation = tf.nn.relu_layer(input_op, kernel, biases, scope)
        p += [kernel, biases]
        return activation

def mpool_op(input_op,name,kh,kw,dh,dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1,kh,kw,1],
                          strides=[1,dh,dw,1],
                          padding='SAME',
                          name=name)



def time_tensorflow_run(session, target, feed,info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    toatl_duration_squared = 0.0

    for i in range(num_batchs + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target,feed_dict = feed)
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
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,
                                               image_size,
                                               image_size,3],
                                              dtype=tf.float32,
                                              stddev=0.1))
        keep_prob = tf.placeholder(tf.float32)
        predictions,softmax,fc8,p = inference_op(images,keep_prob)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        time_tensorflow_run(sess,predictions,{keep_prob:1.0},'Forward')

        objective = tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective,p)
        time_tensorflow_run(sess,grad,{keep_prob:0.5},'Forward-backward')

run_benchmark()