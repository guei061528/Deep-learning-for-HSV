import os
import numpy as np
import cv2
import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from Signaturefunctions.model import *
from Signaturefunctions.CreateICDAR2011_WDdata import *
from Signaturefunctions.functions import *


#Get all Users ID
Train_path = '/home/gliance597/Guei_Project/Python/Signature_Recognition/DataSet/sigComp2011-Chinese-DataSet/Genuine'
train_users = os.listdir(Train_path)
Save_model_path = '/home/gliance597/Guei_Project/Python/Signature_Recognition/network_parameters/Siamese_chinese_parameter/'
#Set train, CV, test samples
train_G_samples = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
train_F_samples = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
CV_G_samples = ['16', '17', '18', '19']
CV_F_samples = ['20', '21', '22', '23', '24']
test_G_samples = ['20', '21', '22', '23']
test_F_samples = ['25', '26', '27', '28', '29']
path = '/home/gliance597/Guei_Project/Python/Signature_Recognition/DataSet/sigComp2011-Chinese-DataSet'
for users in train_users:
    print("User ID = ", users)
    best_org = best_ICDAR2011_org(path, users, train_G_samples)
    print("best user G sample org is ", best_org)
    userData = {
        'filepath': '/home/gliance597/Guei_Project/Python/Signature_Recognition/DataSet/sigComp2011-Chinese-DataSet',
        'UserID': users,
        'train_G_samples': train_G_samples,
        'train_F_samples': train_F_samples,
        'CV_G_samples': CV_G_samples,
        'CV_F_samples': CV_F_samples,
        'best_org': best_org
    }
    all_data = getSiameseTrainData(userData)
    img_org = image_ICDAR_reference(userData['filepath'], users, best_org)
    training_data, training_data_label = all_data['output_train'], all_data['output_train_label']
    CV_data, CV_data_label = all_data['output_CV'], all_data['output_CV_label']
    print("training data and label shape = ", training_data.shape, training_data_label.shape)
    print("CV data and label shape = ", CV_data.shape, CV_data_label.shape)

    tf.reset_default_graph()
    left_org = tf.placeholder(tf.float32, [None, 40960], name='left_org')
    right_org = tf.placeholder(tf.float32, [None, 40960], name='right_org')
    left = tf.reshape(left_org, [-1, 128, 320, 1], name='left')
    right = tf.reshape(right_org, [-1, 128, 320, 1], name='right')
    with tf.name_scope("similarity"):
        label = tf.placeholder(tf.int32, [None, 1], name='label')  # 1 if same, 0 if different
        label = tf.to_float(label)
    
    left_output = Similarnet(left, reuse=False)
    right_output = Similarnet(right, reuse=True)
    margin = 0.2
    loss = contrastive_loss(left_output, right_output, label, margin)
    dis = distance(left_output, right_output)
    global_step = tf.Variable(0, trainable=False)
    train_step = tf.train.AdamOptimizer(0.00001).minimize(loss, global_step=global_step)
    train_size = training_data.shape[0]
    batch_size = 32
    CV_loss_old = 0
    count_up = 0
    count_down = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1200):
            if count_up > 3:
                break
            # if count_down > 3:
            #     break
            batch_mask = np.random.choice(train_size, batch_size, replace=False)
            x_batch = training_data[batch_mask]
            y_batch = training_data_label[batch_mask]
            sess.run(train_step, feed_dict={left_org: img_org, right_org: x_batch, label: y_batch})
            if i % 50 == 0 and i >= 250:
                print("step = ", i)
                Train_loss = sess.run(loss, feed_dict={left_org: img_org, right_org: x_batch, label: y_batch})
                CV_loss = sess.run(loss, feed_dict={left_org: img_org, right_org: CV_data, label: CV_data_label})
                print("train loss =", Train_loss)
                print("CV loss =", CV_loss)
                print("CV distance = ", sess.run(dis, feed_dict={left_org: img_org, right_org: CV_data}))
                if (CV_loss - CV_loss_old) > 0:
                    count_up = count_up + 1
                    count_down = 0
                    print("Number of rises ", count_up)
                else:
                    count_down = count_down + 1
                    count_up = 1
                    print("Number of falls ", count_down)
                CV_loss_old = CV_loss
        saver = tf.train.Saver()
        save_path = saver.save(sess, Save_model_path + users + '/save_Siamese' + users + '.ckpt')


tf.logging.set_verbosity(old_v)
