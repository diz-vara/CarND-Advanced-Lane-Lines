# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 22:00:35 2017

@author: diz
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    if (len(img.shape) == 3 and img.shape[2] == 3):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if (orient == 'x'):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1            
    # 6) Return this mask as your binary_output image
    binary_output =sxbinary
    return binary_output
    

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    if (len(img.shape) == 3 and img.shape[2] == 3):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    # 2) Take the gradient in x and y separately
    sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    abs_sobel = np.sqrt(sobelX*sobelX + sobelY*sobelY)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    sobel8 = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a binary mask where mag thresholds are met
    sbinary = np.zeros_like(sobel8)
    sbinary[(sobel8 >= mag_thresh[0]) & (sobel8 <= mag_thresh[1])] = 1  
    # 6) Return this mask as your binary_output image
    return sbinary
    
def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # 1) Convert to grayscale
    if (len(img.shape) == 3 and img.shape[2] == 3):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    sobelX = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobelY = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    dir_sobel = np.arctan2(sobelY, sobelX)
    # 5) Create a binary mask where direction thresholds are met
    sbinary = np.zeros_like(dir_sobel)
    sbinary[(dir_sobel >= thresh[0]) & (dir_sobel <= thresh[1])] = 1  
    # 6) Return this mask as your binary_output image
    return sbinary
    
    # 6) Return this mask as your binary_output image
    binary_output = np.copy(img) # Remove this line
    return binary_output
#%%

def pipeline(image, s_thresh=(170, 255), sx_thresh=(60, 255)):
    img = np.copy(image)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[ ((s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1]))] = 1
    # Stack each channel
    
    
    
    m_binary = mag_thresh(l_channel,mag_thresh=(100,256), sobel_kernel=11);
    d_binary = dir_thresh(l_channel, sobel_kernel = 15, thresh=(0.8,1.3));
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    #color_binary = np.dstack(( ( (m_binary == 1) & (d_binary == 1))*0.5, sxbinary*0.5, s_binary*0.5))
    result = ( ((m_binary == 1) & (d_binary == 1)) | (sxbinary == 1) | (s_binary==1))
    return result
    
#%%
def rescale2width(img, newWidth):
    scale = newWidth/img.shape[1]
    out = cv2.resize(img, (0,0), fx=scale, fy=scale)
    return out,scale
    
def oldpipe (img, thresh = (10,255)):
    oldH = img.shape[0]
    oldW = img.shape[1]

    r_channel = img[:,:,0];
    #r_channel, scale = rescale2width(r_channel, 480)
    blurred = cv2.medianBlur(r_channel,25);
    diff = cv2.subtract(r_channel, blurred)
    #diff = diff*255//np.max(diff)
    binary = np.zeros_like(diff)
    binary[ ((diff >= thresh[0]) & (diff <= thresh[1]))] = 1
    out = binary
    #out = cv2.resize(binary, (oldW, oldH),cv2.INTER_NEAREST)
    #m_binary = mag_thresh(diff,mag_thresh=(100,256), sobel_kernel=11);
    #d_binary = dir_thresh(diff, sobel_kernel = 15, thresh=(0.8,1.3));

    #result = ( ((m_binary == 1) & (d_binary == 1)) | (binary == 1) )

    return out
    
    
#%%    
def stack(image, s_thresh=(100, 255), sx_thresh=(60, 255)):
    img = np.copy(image)
    # Convert to HSV color space and separate the V channel
    r_channel = img[:,:,0];
    blurred = cv2.medianBlur(r_channel,23);
    s_channel = cv2.subtract(r_channel, blurred)


    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    # Sobel x
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    sobely = cv2.Sobel(s_channel, cv2.CV_64F, 0, 1) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[ ((s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1]))] = 1
    # Stack each channel
    
    m_binary = mag_thresh(s_channel,mag_thresh=(100,256), sobel_kernel=11);
    d_binary = dir_thresh(s_channel, sobel_kernel = 15, thresh=(0.8,1.3));
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( ( (m_binary == 1) & (d_binary == 1))*0.5, sxbinary*0.5, s_binary))
    #result = ( ((m_binary == 1) & (d_binary == 1)) | (sxbinary == 1) | (s_binary==1))
    return color_binary
    
#%%
def threshold_image(image):
    
    rr = oldpipe(image, (25,255))
    #combine



    r = (rr*255).astype(np.uint8)

   
    rwb = np.zeros_like(r)
    rwb[r > 2] = 255
    
    #remove noise
    rwbc = cv2.morphologyEx(rwb, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    #rwbc = rwb
   
    return rwbc
        