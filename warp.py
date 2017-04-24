# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 22:54:00 2017

@author: diz
"""

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "mtx_dist.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

image0 = cv2.imread('test_images/straight_lines1.jpg')

image = cv2.undistort(image0, mtx, dist, None, mtx)

o, lines = detectLanes(image,mode)
height = image.shape[0]

src = np.float32(
   [[ lines[0][0], lines[0][1]],
    [ lines[0][2], lines[0][3]], 
    [ lines[1][0], lines[1][1]],
    [ lines[1][2], lines[1][3]]])

dst = np.float32(
   [[ lines[0][0], height],
    [ lines[0][0], 0], 
    [ lines[1][0], height], 
    [ lines[1][0],  0]])

M = cv2.getPerspectiveTransform(src, dst)
            # e) use cv2.warpPerspective() to warp your image to a top-down view
warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), 
                             flags=cv2.INTER_LINEAR)
cv2.imwrite("warped.png",warped)

pickle.dump(M, open("M.p", "wb"))
