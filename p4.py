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
def rescale2width(img, newWidth):
    scale = newWidth/img.shape[1]
    out = cv2.resize(img, (0,0), fx=scale, fy=scale)
    return out,scale
    

    

     
#%%

def threshold_image(image):
    
    rr = pipeline(image, (90,255))
    #combine



    r = (rr*255).astype(np.uint8)

   
    rwb = np.zeros_like(r)
    rwb[r > 2] = 255
    
    #remove noise
    rwbc = cv2.morphologyEx(rwb, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    #rwbc = rwb
   
    return rwbc
    
def process_image(image):
    global mask;

    oldH = image.shape[0]
    oldW = image.shape[1]
    #res, r_left, r_right = find_and_fit(rwbc)
    dst = cv2.undistort(image, mtx, dist, None, mtx)
    dw = cv2.warpPerspective(dst, M, (oldW, oldH), 
                                 flags=cv2.INTER_LINEAR)
    
    rwbc = threshold_image(dw)
    
    if (not 'mask' in globals()):
        mask = np.zeros_like(rwbc)
    
    res, searchImg, mask = window_fit(rwbc,mask)
    newwarp = cv2.warpPerspective(res, Minv, (oldW, oldH))
    
    result = cv2.addWeighted(dst, 0.8, newwarp, 0.3, 0)
    return result
    
def prc(image):
    oldH = image.shape[0]
    oldW = image.shape[1]
    #res, r_left, r_right = find_and_fit(rwbc)
    dst = cv2.undistort(image, mtx, dist, None, mtx)
    dw = cv2.warpPerspective(dst, M, (oldW, oldH), 
                                 flags=cv2.INTER_LINEAR)
    
    st = stack(dw, (80,255))
    return st   

#%%

left_fit = np.zeros(3)
right_fit = np.zeros(3)
plt.imshow(process_image(images[1]))



#%%

from moviepy.editor import VideoFileClip
#from IPython.display import HTML
    
left_fit = np.zeros(3)
right_fit = np.zeros(3)

video_output = 'out/first.mp4'
clip2 = VideoFileClip('project_video.mp4')
first_clip = clip2.fl_image(process_image)
get_ipython().magic('time first_clip.write_videofile(video_output, audio=False)')






        