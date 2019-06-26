# Fixed to 128, 320

import numpy as np
import cv2
import os
import time
from Signaturefunctions.find_4_corners_15 import *
Btime = time.time()

# Data Path
mypath = "/home/gliance597/Guei_Project/Python/Signature_Recognition/DataSet/sigComp2011-Chinese-DataSet" 

Users = 10
Samples = 30
for i in range(0, Users, 1):
    si = str(i)
    for j in range(0, Samples, 1):
        sj = str(j)
        if os.path.exists(mypath + "/Genuine/"+ si +"/gtnb_" + sj + ".png"):
            pass
        else:
            try:
            	# Read signature image in a gray-scale.
                img = cv2.imread(mypath + "/Genuine/"+ si + "/" + sj + ".png", 0)
                # First binarization
                ret, gray = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                print("gray shape is ", gray.shape)
                # Find Corners by Harris Algorithm
                corners = cv2.goodFeaturesToTrack(gray,5000,0.01,4)
                corners = np.int32(corners)
                corners = corners.reshape(corners.shape[0],2)
                for item in corners:
                    x = item[0]
                    y = item[1]
                    gray[y,x] = 100
                # Calculate the vertex coordinates of the signature
                down_point = compute_down_point(gray)
                left_point = compute_left_point(gray)
                upper_point = compute_upper_point(gray)
                right_point = compute_right_point(gray)

                pts1 = np.float32([[left_point, upper_point], [right_point, upper_point], [left_point, down_point],
                                   [right_point, down_point]])
                pts2 = np.float32([[0, 0], [320, 0], [0, 128], [320, 128]])

                # Geometric transformation to a fixed size
                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(img, M, (320, 128))
                # Second binarization
                ret2, img_binary = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # Save binarization image
                cv2.imwrite(mypath + "/Genuine/"+ si + "/gtnb_" + sj + ".png", img_binary)
                print("User " + si + " for " + sj +" Times Genuine Finish")
            except:
                print("Can't read image")
                pass

for i in range(0, Users, 1):
    si = str(i)
    for j in range(0, Samples, 1):
        sj = str(j)
        if os.path.exists(mypath + "/Forge/"+ si +"/gtnb_" + sj + ".png"):
            pass
        else:
            try:
                img = cv2.imread(mypath + "/Forge/"+ si + "/" + sj + ".png",0)
                ret, gray = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
                ret2, img_binary = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cv2.imwrite(mypath + "/Forge/"+ si + "/gtnb_" + sj + ".png", img_binary)
                print("User " + si + " for " + sj +" Times Forge Finish")
            except:
                print("Can't read image")
                pass


Etime = time.time()

print(Etime-Btime)
