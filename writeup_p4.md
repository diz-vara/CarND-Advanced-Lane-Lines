##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[distorted]: ./output_images/cal_distorted.png "distorted"
[undist0]: ./output_images/cal_undistorted_0.png "undistorted"

[calibr4]: ./camera_cal/calibration4.jpg "calibr4m"
[calibr5]: ./camera_cal/calibration5.jpg "calibr5m"

[undist1]: ./output_images/cal_undistorted_1.png "undistorted"
[undist3]: ./output_images/cal_undistorted_3.png "undistorted"

[imagea]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines # through # of the file called `calibrate.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. I assumed (according to project instructions) that calibration pattern will be fixed, with 6 rows and 
9 columns of 'inner crosses'. Next assumptions are:
  - that calibration pattern is flat (z == 0 for all corners)
  - that calibration consists of squares with some (arbitrary) side length.
  I assigned to that length a value of 40 - just because in several projects I used calibration pattern
with 40 mm squares.
Array of 54 ( 9 x 6 ) 3d points, formed in that way, was added to the **objPoints** list for each image,
in which `cv2.findChessboardCorners` function idendified all the corners.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

After the first run, it became obvious, that:
- corners were not found in images 1,4 and 5
- after uhdistortion, left vertical line in the undistorted image was not straight - it was curved 'outside'
(in the direction opposite to that in original image):

![alt text][distorted]  
![alt text][undist0]


Then I attempted to use different pattern size in cases where `cv2.findChessboardCorners` failed (file `calibration.py`, lines 51-60. For that purpose, I refactored the creation of pattern's object coordinates into
separate fucntion (lines 16-23 of the same file). It helped with the first image, but not with images 4 and 5. In ordinary situation
I would live them alone, and asked user to shot additional images, reminding her that all 6x9 inner crosses
are to be visible in every shot. But here I can't make new shots. And I need these images, because they are
'close-ups', with more prominent distortion.

As the source of the error was the irreqularity of detected patterns (different number of corners in different
rows or columns, I edited these images, wiping-out portions of them. Here are two modified images:

![alt text][calibr4] 

![alt text][calibr5]

After this modification all 20 calibration images were processed, but undistorted images remained curved.

Then I noticed, that by default `cv2.calibrateCamera()` calculates only 5 'basic' distortion coefficients -
and do not calculate coefficients for more advanced camera models. I turned corresponding flags on, one by 
one - and all models were included. Addition of 'Rational model' improved things a bit. With all tilted
model lines became straight - but additional perspective distortion was added (third image).
I think, we need more data to calculate these coefficients.
Unfortunately , in curent version of OpenCV thing prism model without tilted model does not work.
So, I decided to use rational model (8 distortion coefficients, second image).

![alt text][undist0]
![alt text][undist1]
![alt text][undist3]




###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

