# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 11:04:22 2017

@author: Anton Varfolomeev
"""

#%%
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

#%%
os.chdir('D:\\WORK\\CarND\\p4\\CarND-P4')
cal_dir = './camera_cal/';
test_dir = './test_images'
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

#%%

mtxdist = pickle.load(open('mtx_dist.p','rb'))
mm = pickle.load(open('M.p','rb'))
M = mm[0]
Minv = mm[1]

mtx = mtxdist['mtx']
dist=mtxdist['dist']

#%%

images = []
for entry in os.scandir(test_dir):
    if entry.is_file():
        print(entry.path)
        img = cv2.imread(entry.path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        images.append(img)

#%%
img = images[8]

oldH = img.shape[0]
oldW = img.shape[1]

dst = cv2.undistort(img, mtx, dist, None, mtx)
result = pipeline(dst)
#combine
r = (result*255).astype(np.uint8)

#unnwrap
rw = cv2.warpPerspective(r, M, (r.shape[1], r.shape[0]), 
                             flags=cv2.INTER_LINEAR)

rwb = np.zeros_like(rw)
rwb[rw > 2] = 255

#remove noise
rwbc = cv2.morphologyEx(rwb, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))

res, left, right = cluster_fit(rwbc)

cw = cv2.resize(res, (oldW,oldH))
newwarp = cv2.warpPerspective(cw, Minv, (oldW, oldH))

result = cv2.addWeighted(dst, 0.8, newwarp, 0.3, 0)
plt.imshow(result)

#%%



#%%
drawingMode=BOTH
center = Point(0.5, 0.6)
yellow_output = 'out/yellow.mp4'
clip2 = VideoFileClip('project_video.mp4')
yellow_clip = clip2.fl_image(process_image)
get_ipython().magic('time yellow_clip.write_videofile(yellow_output, audio=False)')






        