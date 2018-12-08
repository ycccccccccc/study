import tensorflow as tf
import numpy as np
import readdata
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically


batch_size = 32
num_batchs = 100
class_num = 10
max_step = 10000

model_path = '../../model/Alexnet/'
model_name = 'model.ckpt'

keep_prob = tf.placeholder(tf.float32, None, name='keep_prob')
x = tf.placeholder(tf.float32, [None, None, None, 3], name='x')
y_ = tf.placeholder(tf.float32, [None, class_num], name='y_')


def inference(image):
    image_reshape = tf.image.resize_images(image,[224,224])
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64],
                                                 dtype=tf.float32, stddev=0.1), name='weight')
        conv = tf.nn.conv2d(image_reshape, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[64]),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)

    lrn1 = tf.nn.lrn(conv1, 4, 1.0, 0.001 / 9, 0.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool1')

    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192],
                                                 dtype=tf.float32, stddev=0.1), name='weight')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[192]),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)

    lrn2 = tf.nn.lrn(conv2, 4, 1.0, 0.001 / 9, 0.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool2')

    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                 dtype=tf.float32, stddev=0.1), name='weight')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[384]),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)

    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 dtype=tf.float32, stddev=0.1), name='weight')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256]),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)

    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                 dtype=tf.float32, stddev=0.1), name='weight')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256]),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)

    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool5')

    with tf.name_scope('fc1') as scope:
        flat = tf.reshape(pool5, [-1, 6 * 6 * 256])
        weights = tf.Variable(tf.truncated_normal([6 * 6 * 256, 4096], dtype=tf.float32,
                                                  stddev=0.01), name='weight')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[4096]),
                             trainable=True, name='biases')
        fc1 = tf.nn.dropout(tf.nn.relu(tf.nn.xw_plus_b(flat, weights, biases))
                            , keep_prob=keep_prob, name=scope)

    with tf.name_scope('fc2') as scope:
        weights = tf.Variable(tf.truncated_normal([4096, 4096], dtype=tf.float32,
                                                  stddev=0.01), name='weight')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[4096]),
                             trainable=True, name='biases')
        fc2 = tf.nn.dropout(tf.nn.relu(tf.nn.xw_plus_b(fc1, weights, biases))
                            , keep_prob=keep_prob, name=scope)

    with tf.name_scope('fc3') as scope:
        weights = tf.Variable(tf.truncated_normal([4096, 1000], dtype=tf.float32,
                                                  stddev=0.01), name='weight')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[1000]),
                             trainable=True, name='biases')
        fc3 = tf.nn.dropout(tf.nn.relu(tf.nn.xw_plus_b(fc2, weights, biases))
                            , keep_prob=keep_prob, name=scope)

    with tf.name_scope('classifier') as scope:
        weights = tf.Variable(tf.truncated_normal([1000, class_num], dtype=tf.float32,
                                                  stddev=0.01), name='weight')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[class_num]),
                             trainable=True, name='biases')
        classifier = tf.nn.xw_plus_b(fc3, weights, biases,name=scope)
    return classifier


y = inference(x)
out = tf.argmax(y,1,name='out')
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


def test(sess, times, step, size=0):
    acc_add = 0
    loss_add = 0
    for i in range(times):
        images, labels = readdata.test(size)
        images = np.reshape(images, [-1, 224, 224, 3])
        feed_dict = {x: images, y_: labels, keep_prob: 0.5}
        loss_, acc_ = sess.run([loss, acc], feed_dict)
        loss_add += loss_
        acc_add+=acc_
    print('after %d setp,on test data, loss is %.3f,accuracy is %g' % (step, loss_add/times, acc_add/times))


def train(ctn=True):
    with tf.Session() as sess:
        sess.run(init)
        start_step = 0
        if ctn:
            module_file = tf.train.latest_checkpoint(model_path)
            saver.restore(sess, module_file)
            start_step = int(module_file.split('-')[1])

        for step in range(start_step, max_step):
            images, labels = readdata.batch(batch_size)
            images = np.reshape(images, [-1, 224, 224, 3])
            feed_dict = {x: images, y_: labels, keep_prob: 0.5}
            if step % 10 == 0:
                loss_, acc_ = sess.run([loss, acc], feed_dict)
                print('After %d setp, loss is %.3f,accuracy is %g' % (step, loss_, acc_))
            sess.run(train_step, feed_dict)

            if step % 100 == 0:
                test(sess, 10, step, batch_size)
                saver.save(sess, save_path=model_path + model_name, global_step=step)


train(False)

# with tf.Session() as sess:
#     sess.run(init)
#     start_step = 0
#     module_file = tf.train.latest_checkpoint(model_path)
#     saver.restore(sess, module_file)
#     start_step = int(module_file.split('-')[1])
#
#     tf.saved_model.simple_save(sess, "./model",inputs={"x": x, }, outputs={"out": out, })