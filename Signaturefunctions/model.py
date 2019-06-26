import tensorflow as tf
import numpy as np


# flags = tf.app.flags
# FLAGS = flags.FLAGS

def drop_layer(input, keep_prob, seed_num, getseed):
    if getseed:
        seed_num = np.random.random_integers(87654321)
        drop = tf.nn.dropout(input, keep_prob=keep_prob, seed=seed_num)
        return drop, seed_num
    else:
        drop = tf.nn.dropout(input, keep_prob=keep_prob, seed=seed_num)
        return drop, seed_num

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def Alexnet(input, keep_prob, seed_num, getseed, reuse=False):
    with tf.variable_scope('Alexnet', reuse=reuse):
        seed_list = []
        weight1 = tf.get_variable('weight1', shape=[11, 11, 1, 64],
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases1 = bias_variable([64])
        conv1 = tf.nn.relu(tf.nn.conv2d(input, weight1, [1, 4, 4, 1], padding='SAME') + biases1)
        lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn1')
        pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        print(pool1.shape)
        # 64,160,96
        weight2 = tf.get_variable('weight2', shape=[5, 5, 64, 256],
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases2 = bias_variable([256])
        conv2 = tf.nn.relu(tf.nn.conv2d(pool1, weight2, [1, 1, 1, 1], padding='SAME') + biases2)
        lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn2')
        pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        print(pool2.shape)
        # 32,80,256
        weight3 = tf.get_variable('weight3', shape=[3, 3, 256, 384], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases3 = bias_variable([384])
        conv3 = tf.nn.relu(tf.nn.conv2d(pool2, weight3, [1, 1, 1, 1], padding='SAME') + biases3)
        
        weight4 = tf.get_variable('weight4', shape=[3, 3, 384, 384],
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases4 = bias_variable([384])
        conv4 = tf.nn.relu(tf.nn.conv2d(conv3, weight4, [1, 1, 1, 1], padding='SAME') + biases4)
        # print(pool4.shape)

        weight5 = tf.get_variable('weight5', shape=[3, 3, 384, 256], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases5 = bias_variable([256])
        conv5 = tf.nn.relu(tf.nn.conv2d(conv4, weight5, [1, 1, 1, 1], padding='SAME') + biases5)
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
        print(pool5.shape)
        # 16,40,256
        W_fc1 = tf.get_variable('W_fc1', shape=[4 * 6 * 256, 1024], initializer=tf.contrib.layers.xavier_initializer())
        biases_fc1 = bias_variable([1024])
        # h_pool4_flat = tf.reshape(pool4, [-1, 15 * 39 * 256])
        h_pool5_flat = tf.reshape(pool5, [-1, 4 * 6 * 256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + biases_fc1)
        h_fc1_drop, seed_num_1 = drop_layer(h_fc1, keep_prob=keep_prob, seed_num=seed_num[0], getseed=getseed)
        seed_list.append(seed_num_1)

        W_fc2 = tf.get_variable('W_fc2', shape=[1024, 1024], initializer=tf.contrib.layers.xavier_initializer())
        biases_fc2 = bias_variable([1024])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + biases_fc2)
        h_fc2_drop, seed_num_2 = drop_layer(h_fc2, keep_prob=keep_prob, seed_num=seed_num[1], getseed=getseed)
        seed_list.append(seed_num_2)

        W_fc3 = tf.get_variable('W_fc3', shape=[1024, 256], initializer=tf.contrib.layers.xavier_initializer())
        biases_fc3 = bias_variable([256])
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + biases_fc3)
        if getseed:
            print(seed_list)
            return h_fc3, seed_list
        else:
            return h_fc3

def Signet(input, keep_prob, seed_num, getseed, reuse=False):
    with tf.variable_scope('Signet', reuse=reuse):
        seed_list = []
        weight1 = tf.get_variable('weight1', shape=[11, 11, 1, 96],
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases1 = bias_variable([96])
        conv1 = tf.nn.relu(tf.nn.conv2d(input, weight1, [1, 1, 1, 1], padding='SAME') + biases1)
        lrn1 = tf.nn.lrn(conv1, 5, bias=1.0, alpha=1e-4, beta=0.75, name='lrn1')
        pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        # print(pool1.shape)
        # 64,160,96
        weight2 = tf.get_variable('weight2', shape=[5, 5, 96, 256],
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases2 = bias_variable([256])
        conv2 = tf.nn.relu(tf.nn.conv2d(pool1, weight2, [1, 1, 1, 1], padding='SAME') + biases2)
        lrn2 = tf.nn.lrn(conv2, 5, bias=1.0, alpha=1e-4, beta=0.75, name='lrn2')
        pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        pool2_drop, seed_num_1 = drop_layer(pool2, keep_prob=keep_prob[0], seed_num=seed_num[0], getseed=getseed)
        seed_list.append(seed_num_1)
        # print(pool2.shape)
        # 32,80,256
        weight3 = tf.get_variable('weight3', shape=[3, 3, 256, 384], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases3 = bias_variable([384])
        conv3 = tf.nn.relu(tf.nn.conv2d(pool2, weight3, [1, 1, 1, 1], padding='SAME') + biases3)
        
        weight4 = tf.get_variable('weight4', shape=[3, 3, 384, 256],
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases4 = bias_variable([256])
        conv4 = tf.nn.relu(tf.nn.conv2d(conv3, weight4, [1, 1, 1, 1], padding='SAME') + biases4)
        # print(pool4.shape)

        weight5 = tf.get_variable('weight5', shape=[3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases5 = bias_variable([256])
        conv5 = tf.nn.relu(tf.nn.conv2d(conv4, weight5, [1, 1, 1, 1], padding='SAME') + biases5)
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
        pool5_drop, seed_num_2 = drop_layer(pool5, keep_prob=keep_prob[1], seed_num=seed_num[1], getseed=getseed)
        seed_list.append(seed_num_2)
        print(pool5.shape)
        # 16,40,256
        W_fc1 = tf.get_variable('W_fc1', shape=[18 * 26 * 256, 1024], initializer=tf.contrib.layers.xavier_initializer())
        biases_fc1 = bias_variable([1024])
        # h_pool4_flat = tf.reshape(pool4, [-1, 15 * 39 * 256])
        h_pool5_flat = tf.reshape(pool5, [-1, 18 * 26 * 256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + biases_fc1)
        h_fc1_drop, seed_num_3 = drop_layer(h_fc1, keep_prob=keep_prob[2], seed_num=seed_num[2], getseed=getseed)
        seed_list.append(seed_num_3)

        W_fc2 = tf.get_variable('W_fc2', shape=[1024, 128], initializer=tf.contrib.layers.xavier_initializer())
        biases_fc2 = bias_variable([128])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + biases_fc2)
        if getseed:
            print(seed_list)
            return h_fc2, seed_list
        else:
            return h_fc2


def Similarnet(input, reuse=False):
    with tf.variable_scope('Similarnet', reuse=reuse):
        weight1 = tf.get_variable('weight1', shape=[3, 3, 1, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases1 = bias_variable([64])
        conv1 = tf.nn.relu(tf.nn.conv2d(input, weight1, [1, 1, 1, 1], padding='SAME') + biases1)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

        weight2 = tf.get_variable('weight2', shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases2 = bias_variable([128])
        conv2 = tf.nn.relu(tf.nn.conv2d(pool1, weight2, [1, 1, 1, 1], padding='SAME') + biases2)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool2')

        weight3 = tf.get_variable('weight3', shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases3 = bias_variable([128])
        conv3 = tf.nn.relu(tf.nn.conv2d(pool2, weight3, [1, 1, 1, 1], padding='SAME') + biases3)
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool3')
        weight4 = tf.get_variable('weight4', shape=[3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases4 = bias_variable([256])
        conv4 = tf.nn.relu(tf.nn.conv2d(pool3, weight4, [1, 1, 1, 1], padding='SAME') + biases4)
        pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool4')
        weight5 = tf.get_variable('weight5', shape=[3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases5 = bias_variable([256])
        conv5 = tf.nn.relu(tf.nn.conv2d(pool4, weight5, [1, 1, 1, 1], padding='SAME') + biases5)
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool5')

        W_fc1 = tf.get_variable('W_fc1', shape=[4 * 10 * 256, 256], initializer=tf.contrib.layers.xavier_initializer())
        biases_fc_1 = bias_variable([256])
        h_pool5_flat = tf.reshape(pool5, [-1, 4 * 10 * 256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + biases_fc_1)
    return h_fc1



def contrastive_loss(model1, model2, y, margin):
    with tf.name_scope("contrastive-loss"):
        d = tf.sqrt(tf.reduce_sum(tf.pow(tf.clip_by_value(model1 - model2, 1e-8, tf.reduce_max(model1 - model2)), 2), 1,keep_dims=True))
        tmp = y * tf.square(d)
        tmp2 = (1 - y) * tf.square(tf.maximum((margin - d), 0))
        # w = tf.square(abs(model1))
        return tf.reduce_mean(tmp + tmp2) / 2

def distance(model1, model2):
    with tf.name_scope("distance"):
        d = tf.sqrt(tf.reduce_sum(tf.pow(tf.clip_by_value(model1 - model2, 1e-8, tf.reduce_max(model1 - model2)), 2), 1))
        return d


def simple_net(input, keep_prob, reuse=False):
    with tf.variable_scope('simple_net', reuse=reuse):
        W_fc1 = tf.get_variable('W_fc1', shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
        biases_fc_1 = bias_variable([256])
        h_fc1 = tf.nn.relu(tf.matmul(input, W_fc1) + biases_fc_1)
        h_fc1 = tf.nn.dropout(h_fc1, keep_prob)
    return h_fc1

def simple_net_128(input, keep_prob, reuse=False):
    with tf.variable_scope('simple_net', reuse=reuse):
        W_fc1 = tf.get_variable('W_fc1', shape=[128, 128], initializer=tf.contrib.layers.xavier_initializer())
        biases_fc_1 = bias_variable([128])
        h_fc1 = tf.nn.relu(tf.matmul(input, W_fc1) + biases_fc_1)
        h_fc1 = tf.nn.dropout(h_fc1, keep_prob)
    return h_fc1

