import sys, os
sys.path.append(os.pardir)
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from Signaturefunctions.model import *
from Signaturefunctions.CreateICDAR2011_WDdata import *
from Signaturefunctions.Siamese_ROC_function import *
from Signaturefunctions.functions import *
import copy

def Normalization(x):
    tmp_ = np.mean(x) - np.std(x)
    x_new = copy.deepcopy(x)
    x_nor = ((x-np.mean(x))/np.std(x))
    for i in range(x.shape[0]):
        if x_nor[i] <= 0:
            x_new[i] = x_new[i] - (tmp_*1.2)
        else:
            x_new[i] = x_new[i] + (tmp_*1.2)
    for i in range(x_new.shape[0]):
        if x_new[i] <= 0:
            x_new[i] = 0
        elif x_new[i] >= 1:
            x_new[i] = 1
        else:
            pass
    return x_new


Train_path = '/home/gliance597/Guei_Project/Python/Signature_Recognition/DataSet/sigComp2011-Chinese-DataSet/Genuine'
train_users = os.listdir(Train_path)
model_path = '/home/gliance597/Guei_Project/Python/Signature_Recognition/network_parameters/Siamese_chinese_parameter/'

train_G_samples = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
train_F_samples = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
CV_G_samples = ['16', '17', '18', '19']
CV_F_samples = ['20', '21', '22', '23', '24']
test_G_samples = ['20', '21', '22', '23']
test_F_samples = ['26', '27', '28', '29']
test_G_len = len(test_G_samples)
test_F_len = len(test_F_samples)

Test_Total_Genuine = []
Test_Total_Forge = []

Test_Total_Genuine_np = []
Test_Total_Forge_np = []

for User_ID in train_users:
    print("User ID = ", User_ID)
    userData = {
        'filepath': '/home/gliance597/Guei_Project/Python/Signature_Recognition/DataSet/sigComp2011-Chinese-DataSet',
        'UserID': User_ID,
        'test_G_samples': test_G_samples,
        'test_F_samples': test_F_samples
    }
    all_Data = getSiameseTestData(userData)
    best_org = best_ICDAR2011_org(userData['filepath'], User_ID, train_G_samples)
    print("best user G sample org is ", best_org)
    img_org = image_ICDAR_reference(userData['filepath'], User_ID, best_org)
    testing_G_data, testing_G_data_label = all_Data['output_test_G'], all_Data['output_test_G_label']
    testing_F_data, testing_F_data_label = all_Data['output_test_F'], all_Data['output_test_F_label']
    print("testing G data and label shape = ", testing_G_data.shape, testing_G_data_label.shape)
    print("testing F data and label shape = ", testing_F_data.shape, testing_G_data_label.shape)
    tf.reset_default_graph()
    with tf.Session() as sess:
        left_org = tf.placeholder(tf.float32, [None, 40960], name='left_org')
        right_org = tf.placeholder(tf.float32, [None, 40960], name='right_org')
        left = tf.reshape(left_org, [-1, 128, 320, 1], name='left')
        right = tf.reshape(right_org, [-1, 128, 320, 1], name='right')
        with tf.name_scope("similarity"):
            label = tf.placeholder(tf.int32, [None, 1], name='label')  # 1 if same, 0 if different
            label = tf.to_float(label)
        left_output = Similarnet(left, reuse=False)
        right_output = Similarnet(right, reuse=True)
        saver = tf.train.Saver()
        saver.restore(sess, model_path + User_ID + "/save_Siamese" + User_ID + ".ckpt")
        dis = distance(left_output, right_output)
        distance_G = sess.run(dis, feed_dict={left_org: img_org, right_org: testing_G_data})
        distance_F = sess.run(dis, feed_dict={left_org: img_org, right_org: testing_F_data})
        distance_G_and_F = np.append(distance_G, distance_F, axis=0)
        print(distance_G_and_F.shape)
        Nor_distance_G_and_F = Normalization(distance_G_and_F)
        Test_Total_Genuine.append('User ID ' + User_ID + ' Distance')
        Test_Total_Genuine.append(distance_G)
        Test_Total_Genuine.append('After Normalization Distance')
        Test_Total_Genuine.append(Nor_distance_G_and_F[0:4])
        Test_Total_Genuine_np.append(Nor_distance_G_and_F[0:4])
        Test_Total_Forge.append('User ID ' + User_ID + ' Distance')
        Test_Total_Forge.append(distance_F)
        Test_Total_Forge.append('After Normalization Distance')
        Test_Total_Forge.append(Nor_distance_G_and_F[4:8])
        Test_Total_Forge_np.append(Nor_distance_G_and_F[4:8])


users = len(train_users)
Test_Total_Genuine_np = np.asarray(Test_Total_Genuine_np).reshape([test_G_len*users])
Test_Total_Genuine_label_np = np.ones([test_G_len*users])
Test_Total_Forge_np = np.asarray(Test_Total_Forge_np).reshape([test_F_len*users])
Test_Total_Forge_label_np = np.zeros([test_F_len*users])
print("Test_Total_Genuine_np shape = ", Test_Total_Genuine_np.shape)
print("Test_Total_Forge_np shape = ", Test_Total_Forge_np.shape)
print(Test_Total_Genuine_np)
print(Test_Total_Forge_np)
path_test = '/home/gliance597/Guei_Project/Python/Signature_Recognition/ROC/model101_Siamese_chinese'
np.save(path_test + "/Test_Total_Genuine_np.npy", Test_Total_Genuine_np)
np.save(path_test + "/Test_Total_Genuine_label_np.npy", Test_Total_Genuine_label_np)
np.save(path_test + "/Test_Total_Forge_np.npy", Test_Total_Forge_np)
np.save(path_test + "/Test_Total_Forge_label_np.npy", Test_Total_Forge_label_np)