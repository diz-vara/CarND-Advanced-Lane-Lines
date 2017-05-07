# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 11:04:14 2017

@author: diz
"""


#%%



def window_fit(imageIn, mask):

    global left_fit
    global right_fit
    global leftPts
    
    global left_fitx
    global right_fitx
    
    if (not 'left_fit' in globals()):
        left_fit = np.zeros(3)
        
    if (not 'right_fit' in globals()):
        right_fit = np.zeros(3)    
    
    theta = 0.5
    
    
    imgH = imageIn.shape[0]
    imgW = imageIn.shape[1]

    search_range = 100
    ym_per_pix = 55/imgH # meters per pixel in y dimension
    xm_per_pix = 3.7/imgW # meters per pixel in x dimension

    out_img = np.dstack((imageIn, imageIn, imageIn))
    
    #empty mask - perform search
    masked = imageIn & mask
    leftPts = cv2.findNonZero(np.uint8(masked == 1))
    rightPts = cv2.findNonZero(np.uint8(masked == 2))
    
    if (leftPts == None):
        leftPts = []

    if (rightPts == None):
        rightPts = []


    good_fit = len(leftPts) > 1e2 and len(rightPts) > 1e2
    

    if (good_fit):
        leftY = leftPts[:,0,1]
        rightY = rightPts[:,0,1]
        if (np.max(leftY) - np.min(leftY) < imgH//3):
            good_fit = False;
            

        if (np.max(rightY) - np.min(rightY) < imgH//3):
            good_fit = False;
            
            
    if (not good_fit):
        print ('new detection')
        
        #if (sum (mask))
    
        hstHalf = np.sum(imageIn[imgH//2:,:], axis = 0)
        
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
    
            
        
    
    #else: 
        #use masked
        #print('masked', len(leftPts), len(rightPts))
        

    # Fit a second order polynomial to each
    leftA = np.array(leftPts)
    rightA = np.array(rightPts)
    
    fill_color = (0,155, 0);

    newLeftFit = np.polyfit(leftA[:,0,1], leftA[:,0,0], 2);
    newRightFit = np.polyfit(rightA[:,0,1], rightA[:,0,0], 2);

    ploty = np.arange(imgH)


    new_left_fitx = newLeftFit[0]*ploty**2 + newLeftFit[1]*ploty + newLeftFit[2]
    new_right_fitx = newRightFit[0]*ploty**2 + newRightFit[1]*ploty + newRightFit[2]


    t = theta
    if (sum( left_fit != 0) == 0):
        t = 1
    else:
        diff = np.mean(np.abs(new_left_fitx - left_fitx));
        if (diff > 100):
            print (diff, "old Left")
            fill_color = (0, 0, 255)
    left_fit = (1-t) * left_fit + t * newLeftFit;
 

    t = theta
    if (sum( right_fit != 0) == 0):
        t = 1
    else:
        diff = np.mean(np.abs(new_right_fitx - right_fitx));
        if (diff > 100):
            print (diff, "old Right")
            t = 0.1;
            fill_color = (0, 0, 255)
    right_fit = (1-t) * right_fit + t * newRightFit;
        
        
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


   
    warp_zero = np.zeros_like(imageIn).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))]).astype(np.int)
    pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))]).astype(np.int)

    pts = np.vstack((pts_right[0], np.flipud(pts_left[0])))
    
    cv2.fillPoly(color_warp, [pts], fill_color);


    #prepare mask for future search
    mask = np.zeros_like(imageIn)


    #left search range
    fitx0 = left_fitx - search_range;
    fitx1 = left_fitx + search_range;
    
    pts_left = np.array([np.transpose(np.vstack([fitx0, ploty]))]).astype(np.int)
    pts_right = np.array([np.transpose(np.vstack([fitx1, ploty]))]).astype(np.int)

    pts = np.vstack((pts_right[0], np.flipud(pts_left[0])))
    cv2.fillPoly(mask, [pts], (1));


    #right search range
    fitx0 = right_fitx - search_range;
    fitx1 = right_fitx + search_range;
    
    pts_left = np.array([np.transpose(np.vstack([fitx0, ploty]))]).astype(np.int)
    pts_right = np.array([np.transpose(np.vstack([fitx1, ploty]))]).astype(np.int)

    pts = np.vstack((pts_right[0], np.flipud(pts_left[0])))
    cv2.fillPoly(mask, [pts], (2));


    out_img = cv2.addWeighted(out_img, 0.8, color_warp, 0.3, 0)
    
    #plt.imshow(color_warp)

    #newwarp = cv2.warpPerspective(color_warp, Minv, (newW, newH))
    #plt.imshow(newwarp)
    
    
    #result = cv2.addWeighted(dst, 0.8, newwarp, 0.3, 0)
    #plt.imshow(result)
    
    
    y_eval = np.max(ploty)
    
    
    # Define conversions in x and y from pixels space to meters
    # Fit new polynomials to x,y in world space
    #left_fit_cr = np.polyfit(left[:,1]*ym_per_pix, left[:,0]*xm_per_pix, 2)
    #right_fit_cr = np.polyfit(right[:,1]*ym_per_pix, right[:,0]*xm_per_pix, 2)
    # Calculate the new radii of curvature
    #left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    #right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')

    return color_warp, out_img, mask#, left_curverad, right_curverad

    
