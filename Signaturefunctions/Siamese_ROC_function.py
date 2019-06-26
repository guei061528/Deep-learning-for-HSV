import numpy as np

# Input Forge Data
def False_Acceptance_Rate(True_label, predict_label, thr):
    for i in range(predict_label.shape[0]):
        if predict_label[i] < thr:
            predict_label[i] = 1
        else:
            predict_label[i] = 0
    tmp = 0
    for i in range(True_label.shape[0]):
        if True_label[i] != predict_label[i]:
            tmp = tmp + 1
    return tmp

# Input Genuine Data
def False_Rejection_Rate(True_label, predict_label, thr):
    for i in range(predict_label.shape[0]):
        if predict_label[i] < thr:
            predict_label[i] = 1
        else:
            predict_label[i] = 0
    tmp = 0
    for i in range(True_label.shape[0]):
        if True_label[i] != predict_label[i]:
            tmp = tmp + 1
    return tmp
# Input Genuine Data
def True_Positive_Rate(True_label, predict_label, thr):
    for i in range(predict_label.shape[0]):
        if predict_label[i] < thr:
            predict_label[i] = 1
        else:
            predict_label[i] = 0
    tmp = 0
    for i in range(True_label.shape[0]):
        if True_label[i] != predict_label[i]:
            tmp = tmp + 1
    return predict_label.shape[0]-tmp

# Input Forge Data
def True_Negative_Rate(True_label, predict_label, thr):
    for i in range(predict_label.shape[0]):
        if predict_label[i] < thr:
            predict_label[i] = 1
        else:
            predict_label[i] = 0
    tmp = 0
    for i in range(True_label.shape[0]):
        if True_label[i] != predict_label[i]:
            tmp = tmp + 1
    return predict_label.shape[0]-tmp

# Input Forge Data
def False_Positive_Rate(True_label, predict_label, thr):
    for i in range(predict_label.shape[0]):
        if predict_label[i] < thr:
            predict_label[i] = 1
        else:
            predict_label[i] = 0
    tmp = 0
    for i in range(True_label.shape[0]):
        if True_label[i] != predict_label[i]:
            tmp = tmp + 1
    return tmp
