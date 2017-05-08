# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 11:04:14 2017

@author: diz
"""


#%%
import numpy as np
import cv2

def sliding_fit (imageIn, search_range):

    imgH = imageIn.shape[0]
    imgW = imageIn.shape[1]

    out_img = np.dstack((imageIn, imageIn, imageIn))


    hstHalf = np.sum(imageIn[imgH//2:imgH-imgH-30,:], axis = 0)
    
    middle = imgW//2
    
    left0 = np.argmax(hstHalf[:middle])
    right0 = np.argmax(hstHalf[middle:])+middle
    
    
    nWindows = 12
    windowH = imgH//nWindows
    
    
    left = left0
    right = right0
    
    minpix = 10
    
    leftPts = []
    rightPts = []

    for win in range(nWindows):
        y0 = imgH - (win+1) * windowH
        y1 = imgH - win * windowH
        left_x0 = left - search_range
        left_x1 = left + search_range
        
        right_x0 = right - search_range
        right_x1 = right + search_range
        
        cv2.rectangle(out_img,(left_x0,y0),(left_x1,y1),(0,255,0), 2) 
        cv2.rectangle(out_img,(right_x0,y0),(right_x1,y1),(0,255,0), 2) 
    
        leftRect = np.uint8(imageIn[y0:y1, left_x0:left_x1])
        rightRect = np.uint8(imageIn[y0:y1, right_x0:right_x1])

       
        good_left_points = cv2.findNonZero(leftRect)
        if (good_left_points != None):
            good_left_points = good_left_points + [left_x0, y0]
            leftPts.extend(good_left_points)
            if len(good_left_points) > minpix:
                left = np.int(np.mean(good_left_points,0)[0,0])
 
        good_right_points = cv2.findNonZero(rightRect)
        if (good_right_points != None):
            good_right_points = good_right_points + [right_x0, y0] 
            rightPts.extend(good_right_points)
            if len(good_right_points) > minpix:        
                right = np.int(np.mean(good_right_points,0)[0,0])
    
    return leftPts, rightPts, out_img


def window_fit(imageIn):

    
    global mask
    
    
    imgH = imageIn.shape[0]
    imgW = imageIn.shape[1]

    y_eval = imgH-5


    search_range = 50
    ym_per_pix = 60/imgH # meters per pixel in y dimension
    xm_per_pix = 3.7/730 #imgW # meters per pixel in x dimension

    
    masked = imageIn & mask
    leftPts = cv2.findNonZero(np.uint8(masked == 1))
    rightPts = cv2.findNonZero(np.uint8(masked == 2))
    
    if (leftPts == None):
        leftPts = []

    if (rightPts == None):
        rightPts = []



    good_fit = len(leftPts) > 1e2 and len(rightPts) > 1e2
    

    #check for too short fits
    if (good_fit):
        leftY = leftPts[:,0,1]
        rightY = rightPts[:,0,1]
        if (np.max(leftY) - np.min(leftY) < imgH//4 and np.max(rightY) - np.min(rightY) < imgH//4):
            good_fit = False;
            

    #empty mask - perform sliding search
          
    if (not good_fit):
        leftPts, rightPts, out = sliding_fit(imageIn, search_range)
        print ('new detection')
        
        

    
    fill_color = (0,155, 0);

    ploty = np.arange(imgH)

    
    lineLeft.new_fit(leftPts, ploty);                              
    lineRight.new_fit(rightPts, ploty);                              
                                  
   
    zero_plane = np.zeros_like(imageIn).astype(np.uint8)
    overlay = np.dstack((zero_plane, zero_plane, zero_plane))

    fill_color = (0,155,0)
    fill_between_lines(overlay, lineLeft.old_x, lineRight.old_x, ploty,
                       fill_color)
    


    #prepare mask for future search
    mask = np.zeros_like(imageIn)

    #left search range
    fitx0 = lineLeft.old_x - search_range;
    fitx1 = lineLeft.old_x + search_range;
    fill_between_lines(mask, fitx0, fitx1, ploty, (1))
   

    #right search range
    fitx0 = lineRight.old_x - search_range;
    fitx1 = lineRight.old_x + search_range;
    fill_between_lines(mask, fitx0, fitx1, ploty, (2))
    
    
    # Calculate the new radii of curvature
    left_rad = lineLeft.radius(y_eval)
    right_rad = lineRight.radius(y_eval) 

    
    #print(left_curverad, 'm', right_curverad, 'm')
    curve_m = (left_rad + right_rad)/2
    center = (imgW - (lineLeft.bottomX() + lineRight.bottomX()))/2 * xm_per_pix;

                 
    
    return overlay, curve_m, center

#%%
def fill_between_lines(img, leftX, rightX,y, color):
    pts_left = np.array([np.transpose(np.vstack([leftX, y]))]).astype(np.int)
    pts_right = np.array([np.transpose(np.vstack([rightX, y]))]).astype(np.int)

    pts = np.vstack((pts_right[0], np.flipud(pts_left[0])))
    cv2.fillPoly(img, [pts], color);
    

#%%


def process_image(image):
    global mask;

    oldH = image.shape[0]
    oldW = image.shape[1]
       
    rwbc = threshold_image(image)

     #res, r_left, r_right = find_and_fit(rwbc)
    dst = cv2.undistort(rwbc, mtx, dist, None, mtx)
    dw = cv2.warpPerspective(dst, M, (oldW, oldH), flags=cv2.INTER_NEAREST)
    
    dw[dw > 2] = 255
    
    if (not 'mask' in globals() or mask == None or mask.shape != rwbc.shape):
        mask = np.zeros_like(dw)
    
    res, radius, center = window_fit(dw)
    newwarp = cv2.warpPerspective(res, Minv, (oldW, oldH))
    
    result = cv2.addWeighted(image, 0.8, newwarp, 0.3, 0)
    
    #result = res;
    
    if (radius < 0):
        direction = 'to the left'
    else:
        direction = 'to the right'
    
    radius = np.abs(radius)
    
    if (radius < 5000):
        text = 'Curve radius: ' + '{:4.0f}'.format(radius) + ' m ' + direction
    else:
        text = 'Straight'
    cv2.putText(result, text, (20,50), cv2.FONT_HERSHEY_DUPLEX, 
                1.5, (0,75,0), 2, cv2.LINE_AA)    
    
    if (center < 0):
        direction = 'to the left'
    else:
        direction = 'to the right'
        
    text = 'Car offset: ' + '{:3.1f}'.format(np.abs(center)) + ' m ' + direction;    
    cv2.putText(result, text, (20,100), cv2.FONT_HERSHEY_DUPLEX, 
                1.5, (0,75,0), 2, cv2.LINE_AA)    


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

    
