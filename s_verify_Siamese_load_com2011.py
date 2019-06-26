import os
import tensorflow as tf
import numpy as np
import cv2
from Signaturefunctions.model import *
from Signaturefunctions.CreateICDAR2011_WDdata import *
from Signaturefunctions.Siamese_ROC_function import *
from Signaturefunctions.functions import *
import copy

path_test = '/home/gliance597/Guei_Project/Python/Signature_Recognition/ROC/model4_Siamese'
Test_Total_Genuine_np = np.load(path_test + "/Test_Total_Genuine_np.npy")
Test_Total_Genuine_label_np = np.load(path_test + "/Test_Total_Genuine_label_np.npy")
Test_Total_Forge_np = np.load(path_test + "/Test_Total_Forge_np.npy")
Test_Total_Forge_label_np = np.load(path_test + "/Test_Total_Forge_label_np.npy")
print("Test_Total_Genuine_data shape", Test_Total_Genuine_np.shape)
print("Test_Total_Forge_data shape", Test_Total_Forge_np.shape)
len_G = len(Test_Total_Genuine_label_np)
len_F = len(Test_Total_Forge_label_np)


TPR = []
FPR = []
FAR = []
FRR = []
TNR = []
for thr in np.arange(0, 1.001, 0.001):
    print("Now thr = ", thr)
    tmp_G = copy.deepcopy(Test_Total_Genuine_np)
    tmp_F = copy.deepcopy(Test_Total_Forge_np)
    TPR_container = True_Positive_Rate(Test_Total_Genuine_label_np, tmp_G, thr)
    TNR_container = True_Negative_Rate(Test_Total_Forge_label_np, tmp_F, thr)
    tmp_F = copy.deepcopy(Test_Total_Forge_np)
    FPR_container = False_Positive_Rate(Test_Total_Forge_label_np, tmp_F, thr)
    tmp_F = copy.deepcopy(Test_Total_Forge_np)
    FAR_container = False_Acceptance_Rate(Test_Total_Forge_label_np, tmp_F, thr)
    tmp_G = copy.deepcopy(Test_Total_Genuine_np)
    FRR_container = False_Rejection_Rate(Test_Total_Genuine_label_np, tmp_G, thr)
    TPR.append(TPR_container / len_G)
    TNR.append(TNR_container / len_F)
    FPR.append(FPR_container / len_F)
    FAR.append(FAR_container / len_F)
    FRR.append(FRR_container / len_G)

with open('/home/gliance597/Guei_Project/Python/Signature_Recognition/ROC/model4_Siamese/Siamese_TPR.txt', 'w') as f_TPR:
    for item in TPR:
        f_TPR.write("%s\n" % item)

with open('/home/gliance597/Guei_Project/Python/Signature_Recognition/ROC/model4_Siamese/Siamese_FPR.txt', 'w') as f_FPR:
    for item in FPR:
        f_FPR.write("%s\n" % item)

with open('/home/gliance597/Guei_Project/Python/Signature_Recognition/ROC/model4_Siamese/Siamese_FAR.txt', 'w') as f_FAR:
    for item in FAR:
        f_FAR.write("%s\n" % item)

with open('/home/gliance597/Guei_Project/Python/Signature_Recognition/ROC/model4_Siamese/Siamese_FRR.txt', 'w') as f_FRR:
    for item in FRR:
        f_FRR.write("%s\n" % item)

with open('/home/gliance597/Guei_Project/Python/Signature_Recognition/ROC/model4_Siamese/Siamese_TNR.txt', 'w') as f_TNR:
    for item in TNR:
        f_TNR.write("%s\n" % item)
