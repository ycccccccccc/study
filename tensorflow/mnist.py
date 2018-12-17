import tensorflow as tf
import numpy as np
from datetime import datetime
import math
import time
import collections

# max_step = 10000
# model_path = '../../model/Res/'
# model_name = 'model.ckpt'
# keep_prob = tf.placeholder(tf.float32, None, name='keep_prob')
# x = tf.placeholder(tf.float32, [None, 224, 224, 3], name='x')
# y_ = tf.placeholder(tf.float32, [None, class_num], name='y_')

slim = tf.contrib.slim

class Block(collections.namedtuple('Block',['scope','unit_fc','args'])):
    'A named tuple describing a ResNet block.'

def subsample(inputs,factor,scope = None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs,[1,1],stride = factor,scope = scope)

def conv2d_same(inputs,num_outputs,kernel_size,stride,scope = None):
    if stride == 1:
        return slim.conv2d(inputs,num_outputs,kernel_size,stride = 1,
                           padding = 'VALID',scope = scope)
    else:
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,[[0,0],[pad_beg,pad_end],[pad_beg,pad_end],[0,0]])
        return slim.conv2d(inputs,num_outputs,kernel_size,stride = stride,
                           padding = 'VALID',scope = scope)

@slim.add_arg_scope
def stack_blocks_dense(net,blocks,outputs_collections=None):
    for block in blocks:
        with tf.variable_scope(block.scope,'block',[net]) as sc:
            for i,unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i+1),values=[net]):
                    unit_depth,unit_depth_bottleneck,unit_stride = unit
                    net = block.unit_fc(net,
                                        depth = unit_depth,
                                        depth_bottleneck = unit_depth_bottleneck,
                                        stride = unit_stride)
            net = slim.utils.collect_named_outputs(outputs_collections,sc.name,net)
    return net

def resnet_arg_scope(is_training = True,
                     weight_decay = 0.0001,
                     batch_norm_decay = 0.997,
                     batch_norm_epsilon = 1e-5,
                     batch_norm_scale = True):
    batch_norm_params = {
        'is_training':is_training,
        'decay':batch_norm_decay,
        'epsilon':batch_norm_epsilon,
        'scale':batch_norm_scale,
        'update_collections':tf.GraphKeys.UPDATE_OPS,
    }
    with slim.arg_scope(
        [slim.conv2d],
        weights_regularizer = slim.l2_regularizer(weight_decay),
        weights_initializer = slim.variance_scaling_initializer(),
        activation_fc = tf.nn.relu,
        normalizer_fn = slim.batch_norm,
        normalizer_params = batch_norm_params) :
        with slim.arg_scope([slim.batch_norm],**batch_norm_params):
            with slim.arg_scope([slim.max_pool2d],padding = 'SAME') as arg_sc:
                return arg_sc







def time_tensorflow_run(session, target,info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    toatl_duration_squared = 0.0

    for i in range(num_batchs + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
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
        image_size = 299
        batch_size = 32
        images = tf.Variable(tf.random_normal([batch_size,
                                               image_size,
                                               image_size,3],
                                              dtype=tf.float32,
                                              stddev=0.1))
        with slim.arg_scope(inception_v3_arg_scope()):
            logits,end_points = inception_v3(images,is_training=False)


        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        time_tensorflow_run(sess,logits,'Forward')

run_benchmark()