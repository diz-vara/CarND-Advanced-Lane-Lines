# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 22:00:35 2017

@author: diz
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def pipeline(image, s_thresh=(100, 255), sx_thresh=(60, 255)):
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
    s_binary[l_channel > 120 | ((s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1]))] = 1
    # Stack each channel
    
    
    
    m_binary = mag_thresh(l_channel,mag_thresh=(100,256), sobel_kernel=11);
    d_binary = dir_thresh(l_channel, sobel_kernel = 15, thresh=(0.8,1.3));
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    #color_binary = np.dstack(( ( (m_binary == 1) & (d_binary == 1))*0.5, sxbinary*0.5, s_binary*0.5))
    result = ( ((m_binary == 1) & (d_binary == 1)) | (sxbinary == 1) | (s_binary==1))
    return result
    
#%%    
def stack(image, s_thresh=(170, 255), sx_thresh=(60, 255)):
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
    s_binary[l_channel > 120 | ((s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1]))] = 1
    # Stack each channel
    
    m_binary = mag_thresh(l_channel,mag_thresh=(100,256), sobel_kernel=11);
    d_binary = dir_thresh(l_channel, sobel_kernel = 15, thresh=(0.8,1.3));
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( ( (m_binary == 1) & (d_binary == 1))*0.5, sxbinary*0.5, s_binary*0.5))
    #result = ( ((m_binary == 1) & (d_binary == 1)) | (sxbinary == 1) | (s_binary==1))
    return color_binary