import numpy as np
import cv2

def train_data(path, User_ID, train_G_samples, train_F_samples):
    train_data = []
    train_data_label = []
    for j in train_G_samples:
        train_in_G = cv2.imread(path + '/Genuine/' + User_ID + '/gtnbb_' + j + '.png', 0)
        train_in_G = np.asarray(train_in_G) / 255
        train_in_G = train_in_G.flatten()
        train_data.append(train_in_G)
        tmp = np.zeros(2)
        tmp[0] = 1
        train_data_label.append(tmp)
    for j in train_F_samples:
        train_in_G = cv2.imread(path + '/Forge/' + User_ID + '/gtnbb_' + j + '.png', 0)
        train_in_G = np.asarray(train_in_G) / 255
        train_in_G = train_in_G.flatten()
        train_data.append(train_in_G)
        tmp = np.zeros(2)
        tmp[1] = 1
        train_data_label.append(tmp)
    train_data = np.asarray(train_data)
    train_data_label = np.asarray(train_data_label)
    return train_data, train_data_label

def Cross_Validation_data(path, User_ID, CV_G_samples, CV_F_samples):
    CV_data = []
    CV_data_label = []
    for j in CV_G_samples:
        CV_data_img = cv2.imread(path + '/Genuine/' + User_ID + '/gtnbb_' + j + '.png',0)
        CV_data_img = np.asarray(CV_data_img) / 255
        CV_data_img = CV_data_img.flatten()
        CV_data.append(CV_data_img)
        tmp = np.zeros(2)
        tmp[0] = 1
        CV_data_label.append(tmp)

    for j in CV_F_samples:
        CV_data_img = cv2.imread(path + '/Forge/' + User_ID + '/gtnbb_' + j + '.png',0)
        CV_data_img = np.asarray(CV_data_img) / 255
        CV_data_img = CV_data_img.flatten()
        CV_data.append(CV_data_img)
        tmp = np.zeros(2)
        tmp[1] = 1
        CV_data_label.append(tmp)
    CV_data = np.asarray(CV_data)
    CV_data_label = np.asarray(CV_data_label)
    return CV_data, CV_data_label


def test_Genuine_data(path, User_ID, test_G_samples):
    test_data = []
    test_data_label = []
    for j in test_G_samples:
        test_in_G = cv2.imread(path + '/Genuine/' + User_ID + '/gtnbb_' + j + '.png',0)
        test_in_G = np.asarray(test_in_G) / 255
        test_in_G = test_in_G.flatten()
        test_data.append(test_in_G)
        tmp = np.zeros(2)
        tmp[0] = 1
        test_data_label.append(tmp)
    test_data = np.asarray(test_data)
    test_data_label = np.asarray(test_data_label)
    return test_data, test_data_label

def test_Forge_data(path, User_ID, test_F_samples):
    test_data = []
    test_data_label = []
    for j in test_F_samples:
        test_in_F = cv2.imread(path + '/Forge/' + User_ID + '/gtnbb_' + j + '.png',0)
        test_in_F = np.asarray(test_in_F) / 255
        test_in_F = test_in_F.flatten()
        test_data.append(test_in_F)
        tmp = np.zeros(2)
        tmp[1] = 1
        test_data_label.append(tmp)
    test_data = np.asarray(test_data)
    test_data_label = np.asarray(test_data_label)
    return test_data, test_data_label


def train_siamese_data(path, User_ID, train_G_samples, train_F_samples, best_org):
    train_data = []
    train_data_label = []
    for j in train_G_samples:
        if j == best_org:
            pass
        else:
            train_in_G = cv2.imread(path + '/Genuine/' + User_ID + '/gtnbb_' + j + '.png', 0)
            train_in_G = np.asarray(train_in_G) / 255
            train_in_G = train_in_G.flatten()
            train_data.append(train_in_G)
            tmp = np.zeros(1)
            tmp[0] = 1
            train_data_label.append(tmp)
    for j in train_F_samples:
        train_in_G = cv2.imread(path + '/Forge/' + User_ID + '/gtnbb_' + j + '.png', 0)
        train_in_G = np.asarray(train_in_G) / 255
        train_in_G = train_in_G.flatten()
        train_data.append(train_in_G)
        tmp = np.zeros(1)
        tmp[0] = 0
        train_data_label.append(tmp)
    train_data = np.asarray(train_data)
    train_data_label = np.asarray(train_data_label)
    return train_data, train_data_label

def CV_siamese_data(path, User_ID, CV_G_samples, CV_F_samples):
    CV_data = []
    CV_data_label = []
    for j in CV_G_samples:
        CV_data_img = cv2.imread(path + '/Genuine/' + User_ID + '/gtnbb_' + j + '.png',0)
        CV_data_img = np.asarray(CV_data_img) / 255
        CV_data_img = CV_data_img.flatten()
        CV_data.append(CV_data_img)
        tmp = np.zeros(1)
        tmp[0] = 1
        CV_data_label.append(tmp)

    for j in CV_F_samples:
        CV_data_img = cv2.imread(path + '/Forge/' + User_ID + '/gtnbb_' + j + '.png',0)
        CV_data_img = np.asarray(CV_data_img) / 255
        CV_data_img = CV_data_img.flatten()
        CV_data.append(CV_data_img)
        tmp = np.zeros(1)
        tmp[0] = 0
        CV_data_label.append(tmp)
    CV_data = np.asarray(CV_data)
    CV_data_label = np.asarray(CV_data_label)
    return CV_data, CV_data_label

def test_siamese_G_data(path, User_ID, test_G_samples):
    test_data = []
    test_data_label = []
    for j in test_G_samples:
        test_in_G = cv2.imread(path + '/Genuine/' + User_ID + '/gtnbb_' + j + '.png',0)
        test_in_G = np.asarray(test_in_G) / 255
        test_in_G = test_in_G.flatten()
        test_data.append(test_in_G)
        tmp = np.zeros(1)
        tmp[0] = 1
        test_data_label.append(tmp)
    test_data = np.asarray(test_data)
    test_data_label = np.asarray(test_data_label)
    return test_data, test_data_label

def test_siamese_F_data(path, User_ID, test_F_samples):
    test_data = []
    test_data_label = []
    for j in test_F_samples:
        test_in_F = cv2.imread(path + '/Forge/' + User_ID + '/gtnbb_' + j + '.png',0)
        test_in_F = np.asarray(test_in_F) / 255
        test_in_F = test_in_F.flatten()
        test_data.append(test_in_F)
        tmp = np.zeros(1)
        tmp[0] = 0
        test_data_label.append(tmp)
    test_data = np.asarray(test_data)
    test_data_label = np.asarray(test_data_label)
    return test_data, test_data_label

#######################################################################################################################################

def train_2ch_siamese_data(path, User_ID, train_G_samples, train_F_samples, best_org):
    train_data = []
    train_data_label = []
    img_ref = cv2.imread(path + '/Genuine/' + User_ID + '/gtnbb_' + best_org + '.png', 0)
    img_ref = np.asarray(img_ref) / 255
    img_ref = np.reshape(img_ref, [128, 320, 1])
    for j in train_G_samples:
        if j == best_org:
            pass
        else:
            train_in_G = cv2.imread(path + '/Genuine/' + User_ID + '/gtnbb_' + j + '.png', 0)
            train_in_G = np.asarray(train_in_G) / 255
            train_in_G = train_in_G.reshape([128, 320, 1])
            train_in_G_2ch = np.concatenate((img_ref, train_in_G), axis=2)
            train_data.append(train_in_G_2ch)
            tmp = np.zeros(1)
            tmp[0] = 1
            train_data_label.append(tmp)
    for j in train_F_samples:
        train_in_F = cv2.imread(path + '/Forge/' + User_ID + '/gtnbb_' + j + '.png', 0)
        train_in_F = np.asarray(train_in_F) / 255
        train_in_F = train_in_F.reshape([128, 320, 1])
        train_in_F_2ch = np.concatenate((img_ref, train_in_F), axis=2)
        train_data.append(train_in_F_2ch)
        tmp = np.zeros(1)
        tmp[0] = 0
        train_data_label.append(tmp)
    train_data = np.asarray(train_data)
    train_data_label = np.asarray(train_data_label)
    return train_data, train_data_label

def CV_2ch_siamese_data(path, User_ID, cv_G_samples, cv_F_samples, best_org):
    CV_data = []
    CV_data_label = []
    img_ref = cv2.imread(path + '/Genuine/' + User_ID + '/gtnbb_' + best_org + '.png',0)
    img_ref = np.asarray(img_ref) / 255
    img_ref = np.reshape(img_ref, [128, 320, 1])
    for j in cv_G_samples:
        CV_data_img = cv2.imread(path + '/Genuine/' + User_ID + '/gtnbb_' + j + '.png',0)
        CV_data_img = np.asarray(CV_data_img) / 255
        CV_data_img = CV_data_img.reshape([128, 320, 1])
        CV_2ch_data_img = np.concatenate((img_ref, CV_data_img), axis=2)
        CV_data.append(CV_2ch_data_img)
        tmp = np.zeros(1)
        tmp[0] = 1
        CV_data_label.append(tmp)

    for j in cv_F_samples:
        CV_data_img = cv2.imread(path + '/Forge/' + User_ID + '/gtnbb_' + j + '.png',0)
        CV_data_img = np.asarray(CV_data_img) / 255
        CV_data_img = CV_data_img.reshape([128, 320, 1])
        CV_2ch_data_img = np.concatenate((img_ref, CV_data_img), axis=2)
        CV_data.append(CV_2ch_data_img)
        tmp = np.zeros(1)
        tmp[0] = 0
        CV_data_label.append(tmp)
    CV_data = np.asarray(CV_data)
    CV_data_label = np.asarray(CV_data_label)
    return CV_data, CV_data_label


def test_2ch_siamese_G_data(path, User_ID, test_G_samples, best_org):
    test_data = []
    test_data_label = []
    img_ref = cv2.imread(path + '/Genuine/' + User_ID + '/gtnbb_' + best_org + '.png', 0)
    img_ref = np.asarray(img_ref) / 255
    img_ref = np.reshape(img_ref, [128, 320, 1])
    for j in test_G_samples:
        test_in_G = cv2.imread(path + '/Genuine/' + User_ID + '/gtnbb_' + j + '.png',0)
        test_in_G = np.asarray(test_in_G) / 255
        test_in_G = test_in_G.reshape([128, 320, 1])
        test_2ch_in_G = np.concatenate((img_ref, test_in_G), axis=2)
        test_data.append(test_2ch_in_G)
        tmp = np.zeros(1)
        tmp[0] = 1
        test_data_label.append(tmp)
    test_data = np.asarray(test_data)
    test_data_label = np.asarray(test_data_label)
    return test_data, test_data_label

def test_2ch_siamese_F_data(path, User_ID, test_F_samples, best_org):
    test_data = []
    test_data_label = []
    img_ref = cv2.imread(path + '/Forge/' + User_ID + '/gtnbb_' + best_org + '.png', 0)
    img_ref = np.asarray(img_ref) / 255
    img_ref = np.reshape(img_ref, [128, 320, 1])
    for j in test_F_samples:
        test_in_F = cv2.imread(path + '/Forge/' + User_ID + '/gtnbb_' + j + '.png',0)
        test_in_F = np.asarray(test_in_F) / 255
        test_in_F = test_in_F.reshape([128, 320, 1])
        test_2ch_in_F = np.concatenate((img_ref, test_in_F), axis=2)
        test_data.append(test_2ch_in_F)
        tmp = np.zeros(1)
        tmp[0] = 0
        test_data_label.append(tmp)
    test_data = np.asarray(test_data)
    test_data_label = np.asarray(test_data_label)
    return test_data, test_data_label

######################################################################################################################################################

def getTrainData(userData):
    output_train, output_train_label = train_data(userData['filepath'], userData['UserID'], userData['train_G_samples'], userData['train_F_samples'])
    output_CV, output_CV_label = Cross_Validation_data(userData['filepath'], userData['UserID'], userData['CV_G_samples'], userData['CV_F_samples'])

    return {
        'output_train': output_train,
        'output_CV': output_CV,
        'output_train_label': output_train_label,
        'output_CV_label': output_CV_label,
    }

def getTestData(userData):
    output_test_G, output_test_G_label = test_Genuine_data(userData['filepath'], userData['UserID'], userData['test_G_samples'])
    output_test_F, output_test_F_label = test_Forge_data(userData['filepath'], userData['UserID'], userData['test_F_samples'])
    
    return {
        'output_test_G': output_test_G,
        'output_test_F': output_test_F,
        'output_test_G_label': output_test_G_label,
        'output_test_F_label': output_test_F_label
    }

def getSiameseTrainData(userData):
    output_train, output_train_label = train_siamese_data(userData['filepath'], userData['UserID'], userData['train_G_samples'], userData['train_F_samples'], userData['best_org'])
    output_CV, output_CV_label = CV_siamese_data(userData['filepath'], userData['UserID'], userData['CV_G_samples'], userData['CV_F_samples'])

    return {
        'output_train': output_train,
        'output_CV': output_CV,
        'output_train_label': output_train_label,
        'output_CV_label': output_CV_label,
    }


def getSiameseTestData(userData):
    output_test_G, output_test_G_label = test_siamese_G_data(userData['filepath'], userData['UserID'], userData['test_G_samples'])
    output_test_F, output_test_F_label = test_siamese_F_data(userData['filepath'], userData['UserID'], userData['test_F_samples'])
    
    return {
        'output_test_G': output_test_G,
        'output_test_F': output_test_F,
        'output_test_G_label': output_test_G_label,
        'output_test_F_label': output_test_F_label
    }

def get2chSiameseTrainData(userData):
    output_train, output_train_label = train_2ch_siamese_data(userData['filepath'], userData['UserID'], userData['train_G_samples'], userData['train_F_samples'], userData['best_org'])
    output_CV, output_CV_label = CV_2ch_siamese_data(userData['filepath'], userData['UserID'], userData['CV_G_samples'], userData['CV_F_samples'], userData['best_org'])

    return {
        'output_train': output_train,
        'output_CV': output_CV,
        'output_train_label': output_train_label,
        'output_CV_label': output_CV_label,
    }


def get2chSiameseTestData(userData):
    output_test_G, output_test_G_label = test_2ch_siamese_G_data(userData['filepath'], userData['UserID'], userData['test_G_samples'], userData['best_org'])
    output_test_F, output_test_F_label = test_2ch_siamese_F_data(userData['filepath'], userData['UserID'], userData['test_F_samples'], userData['best_org'])
    
    return {
        'output_test_G': output_test_G,
        'output_test_F': output_test_F,
        'output_test_G_label': output_test_G_label,
        'output_test_F_label': output_test_F_label
    }
#
# userData = {
#     'filepath': 'F:/Guei_Project/Python/Signature_Recognition/sigComp2011-DataSet',
#     'UserID': User_ID,
#     'train_G_samples': 16,
#     'train_F_samples': 4,
#     'CV_G_samples':
#     'CV_F_samples':
#     'test_G_samples': 4,
#     'test_F_samples': 4
# }
# print(type(userData['UserID']))
#
# getAllData(userData)
#
#     userData = {
#         'filepath': 'F:/Guei_Project/Python/Signature_Recognition/sigComp2011-DataSet',
#         'UserID': User_ID,
#         'train_samples': 16,
#         'cv_samples': 4,
#         'test_G_samples': 4,
#         'test_F_samples': 4
#     }
#     all_data = getAllData(userData)