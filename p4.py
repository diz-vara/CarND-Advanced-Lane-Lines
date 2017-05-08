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
#runfile('calibrate.py')
#ret, mtx, dist = calibrate(cal_dir,9,6)
#pickle.dump({"mtx":mtx, "dist":dist}, open("mtx_dist.p", "wb"))

#%%
#read distorted image, display original and undistorted

#this image was not used for calibration 
# (not all rows visible)
#img = cv2.imread(cal_dir + '/calibration1.jpg')
#dst = cv2.undistort(img, mtx, dist, None, None)

#cv2.imwrite(out_dir + 'cal_distorted.png', img)
#cv2.imwrite(out_dir + 'cal_undistorted.png', dst)

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


    

     

#%%

left_fit = np.zeros(3)
right_fit = np.zeros(3)
leftPts = []
rightPts=[]
mask=None
plt.imshow(process_image(images[8]))



#%%

from moviepy.editor import VideoFileClip
#from IPython.display import HTML
     
left_fit = np.zeros(3)
right_fit = np.zeros(3)
leftPts = [] 
rightPts = []

video_output = 'out/old_4.mp4'
clip2 = VideoFileClip('project_video.mp4')
first_clip = clip2.fl_image(process_image)
get_ipython().magic('time first_clip.write_videofile(video_output, audio=False)')



#%%
from moviepy.editor import VideoFileClip
#from IPython.display import HTML
     
left_fit = np.zeros(3)
right_fit = np.zeros(3)
leftPts = [] 
rightPts = []

video_output = 'out/old_chal.mp4'
clip2 = VideoFileClip('challenge_video.mp4')
first_clip = clip2.fl_image(process_image)
get_ipython().magic('time first_clip.write_videofile(video_output, audio=False)')




        