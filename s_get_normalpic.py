# 128,320

import numpy as np
import cv2
import os
import time
from Signaturefunctions.find_4_corners_15 import *
Btime = time.time()

Users = 10000
Samples = 30
for i in range(0, Users, 1):
    si = str(i)
    for j in range(0, Samples, 1):
        sj = str(j)
        if os.path.exists("/home/gliance597/Guei_Project/Python/Signature_Recognition/DataSet/sigComp2011-Chinese-DataSet/Genuine/"+ si +"/gtnb_" + sj + ".png"):
            pass
        else:
            try:
                img = cv2.imread("/home/gliance597/Guei_Project/Python/Signature_Recognition/DataSet/sigComp2011-Chinese-DataSet/Genuine/"+ si + "/" + sj + ".png",0)
                ret, gray = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
                print("gray shape is ", gray.shape)
                corners = cv2.goodFeaturesToTrack(gray,5000,0.01,4)
                corners = np.int32(corners)
                corners = corners.reshape(corners.shape[0],2)
                for item in corners:
                    x = item[0]
                    y = item[1]
                    gray[y,x] = 100

                down_point = compute_down_point(gray)
                left_point = compute_left_point(gray)
                upper_point = compute_upper_point(gray)
                right_point = compute_right_point(gray)

                pts1 = np.float32([[left_point, upper_point], [right_point, upper_point], [left_point, down_point],
                                   [right_point, down_point]])
                pts2 = np.float32([[0, 0], [320, 0], [0, 128], [320, 128]])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(img, M, (320, 128))
                cv2.imwrite("/home/gliance597/Guei_Project/Python/Signature_Recognition/DataSet/sigComp2011-Chinese-DataSet/Genuine/"+ si + "/gtnb_" + sj + ".png", dst)
                print("User " + si + " for " + sj +" Times Genuine Finish")
            except:
                print("Can't read image")
                pass

for i in range(0, Users, 1):
    si = str(i)
    for j in range(0, Samples, 1):
        sj = str(j)
        if os.path.exists("/home/gliance597/Guei_Project/Python/Signature_Recognition/DataSet/sigComp2011-Chinese-DataSet/Forge/"+ si +"/gtnb_" + sj + ".png"):
            pass
        else:
            try:
                img = cv2.imread("/home/gliance597/Guei_Project/Python/Signature_Recognition/DataSet/sigComp2011-Chinese-DataSet/Forge/"+ si + "/" + sj + ".png",0)
                ret, gray = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
                print("gray shape is ", gray.shape)
                corners = cv2.goodFeaturesToTrack(gray,5000,0.01,4)
                corners = np.int32(corners)
                corners = corners.reshape(corners.shape[0],2)
                for item in corners:
                    x = item[0]
                    y = item[1]
                    gray[y,x] = 100

                down_point = compute_down_point(gray)
                left_point = compute_left_point(gray)
                upper_point = compute_upper_point(gray)
                right_point = compute_right_point(gray)

                pts1 = np.float32([[left_point, upper_point], [right_point, upper_point], [left_point, down_point],
                                   [right_point, down_point]])
                pts2 = np.float32([[0, 0], [320, 0], [0, 128], [320, 128]])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(img, M, (320, 128))
                cv2.imwrite("/home/gliance597/Guei_Project/Python/Signature_Recognition/DataSet/sigComp2011-Chinese-DataSet/Forge/"+ si + "/gtnb_" + sj + ".png", dst)
                print("User " + si + " for " + sj +" Times Forge Finish")
            except:
                print("Can't read image")
                pass


Etime = time.time()

print(Etime-Btime)