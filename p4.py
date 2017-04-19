# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 11:04:22 2017

@author: Anton Varfolomeev
"""
cal_dir = './camera_cal/';
out_dir = './output_images/'
#%%
runfile('calibrate.py')
ret, mtx, dist = calibrate(cal_dir,9,6)
pickle.dump({"mtx":mtx, "dist":dist}, open("mtx_dist.p", "wb"))

#%%
#read distorted image, display original and undistorted

#this image was not used for calibration 
# (not all rows visible)
img = cv2.imread(cal_dir + '/calibration1.jpg')
dst = cv2.undistort(img, mtx, dist, None, None)

cv2.imwrite(out_dir + 'cal_distorted.png', img)
cv2.imwrite(out_dir + 'cal_undistorted.png', dst)

