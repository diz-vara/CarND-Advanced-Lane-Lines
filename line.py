# -*- coding: utf-8 -*-
"""
Created on Sat May  6 14:12:07 2017

@author: Anton Varfolomeev
"""

#Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, color):
        #base time constant 
        self.tau0 = 0.8
        #time constant
        self.tau = self.tau0
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the fits of the line
        self.recent_x = None 
        #average x values of the fitted line over the last iterations
        self.old_x = None     
        #polynomial coefficients averaged over the last iterations
        self.old_fit = np.zeros(3) 
        #polynomial coefficients for the most recent fit
        self.current_fit = None  
        #fin in meters
        self.fit_in_meters = np.zeros(3)
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        
        #calibration
        self.ym_per_pix = 60/720
        self.xm_per_pix = 3.7/730 #imgW # meters per pixel in x dimension
        #mask color
        self.color = color
        
    def new_fit(self, pts, y):
        if (len(pts) > 0):
            self.detected = True
            points = np.array(pts)[:,0,:]
            self.current_fit = np.polyfit(points[:,1], points[:,0], 2);
            # Fit new polynomials to x,y in world space
            new_fit_in_meters = np.polyfit(points[:,1]*self.ym_per_pix, 
                                         points[:,0]*self.xm_per_pix, 2)
            
            recent_x = self.current_fit[0]*y**2 + self.current_fit[1]*y + self.current_fit[2]
            t = self.tau
            if (sum( self.old_fit != 0) == 0):
                t = 0
            else:
                diff = np.mean(np.abs(recent_x - self.old_x));
                if (diff > 100):
                    t = 0.95
            self.old_fit = (t) * self.old_fit + (1-t) * self.current_fit;
            self.fit_in_meters = (t) * self.fit_in_meters + (1-t) * new_fit_in_meters;
            self.tau = self.tau0 #restore original tau
        else:
            self.detected = False
            self.tau = self.tau * 0.9 #decrease value of old data
    
        self.old_x = self.old_fit[0]*y**2 + self.old_fit[1]*y + self.old_fit[2]
        
    def radius(self, y_eval):
        curve_rad = ((1 + (2*self.fit_in_meters[0]*y_eval*self.ym_per_pix + self.fit_in_meters[1])**2)**1.5) / (2*self.fit_in_meters[0])
        return curve_rad
        
    #return x-coordinage of the bottom of the line (intercept)    
    def bottomX(self):
        return self.old_x[-1]
