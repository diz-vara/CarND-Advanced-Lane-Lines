# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 11:04:14 2017

@author: diz
"""

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

model = AgglomerativeClustering(n_clusters=2,
                                    linkage="average", affinity="euclidean")

#%%
newW = 480
newH = 270

ym_per_pix = 55/newH # meters per pixel in y dimension
xm_per_pix = 3.7/newW # meters per pixel in x dimension


def cluster_fit(imageIn):

    s=cv2.resize(imageIn,(newW,newH))
    points = cv2.findNonZero(s)
    p = points[:,0,:]
    model.fit(p)


    rightIdx = np.where([model.labels_ == 0])[1]
    leftIdx = np.where([model.labels_ == 1])[1]

    right=p[rightIdx]
    left=p[leftIdx]

    ploty = np.linspace(0, newH-1, num=newH)# to cover same y-range as image


    left_fit = np.polyfit(left[:,1], left[:,0], 2)
    right_fit = np.polyfit(right[:,1], right[:,0], 2)
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    warp_zero = np.zeros_like(s).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))]).astype(np.int)
    pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))]).astype(np.int)

    pts = np.vstack((pts_right[0], np.flipud(pts_left[0])))
    
    cv2.fillPoly(color_warp, [pts], (0,155, 0));
    cv2.polylines(color_warp,pts_left,False,(255,0,0),3);
    cv2.polylines(color_warp,pts_right,False,(0,0,255),3);
    
    #plt.imshow(color_warp)

    #newwarp = cv2.warpPerspective(color_warp, Minv, (newW, newH))
    #plt.imshow(newwarp)
    
    
    #result = cv2.addWeighted(dst, 0.8, newwarp, 0.3, 0)
    #plt.imshow(result)
    
    
    y_eval = np.max(ploty)
    
    
    # Define conversions in x and y from pixels space to meters
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(left[:,1]*ym_per_pix, left[:,0]*xm_per_pix, 2)
    right_fit_cr = np.polyfit(right[:,1]*ym_per_pix, right[:,0]*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')

    return color_warp, left_curverad, right_curverad

    
#%%







#%%
cw = cv2.resize(color_warp, (1280,738))
newwarp = cv2.warpPerspective(cw, Minv, (1280, 738))
plt.imshow(newwarp)


result = cv2.addWeighted(dst, 0.8, newwarp, 0.3, 0)
plt.imshow(result)
