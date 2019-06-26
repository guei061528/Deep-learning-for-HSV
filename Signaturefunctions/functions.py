import numpy as np
import cv2

def best_ICDAR2011_org(path, UserID, train_samples):
    user_score = []
    for i in train_samples:
        tmp = []
        train_org = cv2.imread(path + '/Genuine/' + UserID + "/gtnbb_" + i + ".png", 0)
        gray_org = np.asarray(train_org)
        gray_org = gray_org.flatten()
        for j in train_samples:
            if j == i:
                pass
            else:
                train_ = cv2.imread(path + '/Genuine/' + UserID + '/gtnbb_' + j + '.png', 0)
                gray_ = np.asarray(train_)
                gray_ = gray_.flatten()
                tmp.append(np.count_nonzero(gray_ ^ gray_org))
        user_score.append(sum(tmp)/len(tmp))

    print(user_score)
    best_user = np.argmin(user_score)
    best_user = train_samples[best_user]
    return best_user


def best_gpds_org(path, UserID, train_samples):
    user_score = []
    for i in train_samples:
        tmp = []
        train_org = cv2.imread(path + '/' + UserID + '/c-gtn-' + UserID + '-' + i + '.jpg', 0)
        gray_org = np.asarray(train_org)
        gray_org = gray_org.flatten()
        for j in train_samples:
            if j == i:
                pass
            else:
                train_ = cv2.imread(path + '/' + UserID + '/c-gtn-' + UserID + '-' + j + '.jpg', 0)
                gray_ = np.asarray(train_)
                gray_ = gray_.flatten()
                tmp.append(np.count_nonzero(gray_ ^ gray_org))
        user_score.append(sum(tmp)/len(tmp))

    best_user = np.argmin(user_score)
    best_user = train_samples[best_user]
    return best_user

def best_gpds_org_median(path, UserID, train_samples):
    user_score = np.array([])
    for i in train_samples:
        tmp = []
        train_org = cv2.imread(path + '/' + UserID + '/c-gtn-' + UserID + '-' + i + '.jpg', 0)
        gray_org = np.asarray(train_org)
        gray_org = gray_org.flatten()
        for j in train_samples:
            if j == i:
                pass
            else:
                train_ = cv2.imread(path + '/' + UserID + '/c-gtn-' + UserID + '-' + j + '.jpg', 0)
                gray_ = np.asarray(train_)
                gray_ = gray_.flatten()
                tmp.append(np.count_nonzero(gray_ ^ gray_org))
        user_score = np.append(user_score, sum(tmp)/len(tmp))

    best_user = np.median(user_score)
    best_user = np.where(user_score < best_user)
    best_user_r = np.argmax(best_user)
    best_user_org = train_samples[best_user_r]
    return best_user_org

def image_ICDAR_reference(path, User, best_REF):
    img_ref = cv2.imread(path + '/Genuine/' + User + '/gtnbb_' + best_REF + '.png',0)
    img_ref = np.asarray(img_ref) / 255
    img_ref = np.reshape(img_ref, [1,40960])
    return img_ref

def image_gpds_reference(path, User, best_REF):
    sj = str(best_REF)
    img_ref = cv2.imread(path + '/' + User + '/c-gtn-' + User + '-' + sj + '.jpg', 0)
    img_ref = np.asarray(img_ref) / 255
    img_ref = np.reshape(img_ref, [1,40960])
    return img_ref

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def image_ICDAR_reference_155x220(path, User, best_REF):
    img_ref = cv2.imread(path + '/Genuine/' + User + '/gtnbb_' + best_REF + '.png',0)
    img_ref = np.asarray(img_ref) / 255
    img_ref = np.reshape(img_ref, [1,34100])
    return img_ref


def image_gpds_reference_155x220(path, User, best_REF):
    sj = str(best_REF)
    img_ref = cv2.imread(path + '/' + User + '/c-gtn-' + User + '-' + sj + '.jpg', 0)
    img_ref = np.asarray(img_ref) / 255
    img_ref = img_ref.reshape([1, 34100])
    return img_ref