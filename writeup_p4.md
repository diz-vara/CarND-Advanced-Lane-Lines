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
[undist2]: ./output_images/cal_undistorted_2.png "undistorted"
[undist3]: ./output_images/cal_undistorted_3.png "undistorted"

[imagea]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"


### Camera Calibration



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

Original (distorted) image				

<img src="./output_images/cal_distorted.png" width="500">


Undistorted image

<img src="./output_images/cal_undistorted_0.png" width="500">
 


Then I attempted to use different pattern size in cases where `cv2.findChessboardCorners` failed (file `calibration.py`, lines 51-60. For that purpose, I refactored the creation of pattern's object coordinates into
separate fucntion (lines 16-23 of the same file). It helped with the first image, but not with images 4 and 5. In ordinary situation
I would live them alone, and asked user to shot additional images, reminding her that all 6x9 inner crosses
are to be visible in every shot. But here I can't make new shots. And I need these images, because they are
'close-ups', with more prominent distortion.

As the source of the error was the irreqularity of detected patterns (different number of corners in different
rows or columns, I edited these images, wiping-out portions of them. Here are two modified images:

**Modified calibration4.jpg ( pattern 8x6) and calibration5.jpg ( pattern 9x5)**

<img src="./camera_cal/calibration4.jpg" width="400">
<img src="./camera_cal/calibration5.jpg" width="400">


After this modification all 20 calibration images were processed, but undistorted images remained curved.

Then I noticed, that by default `cv2.calibrateCamera()` calculates only 5 'basic' distortion coefficients -
and do not calculate coefficients for more advanced camera models. I turned corresponding flags on, one by 
one - and all models were included. Addition of 'Rational model' improved things a bit. With all tilted
model lines became straight - but additional perspective distortion was added (third image).
I think, we need more data to calculate these coefficients.
Unfortunately , in curent version of OpenCV thing prism model without tilted model does not work.
So, I decided to use rational model (8 distortion coefficients, second image).

flags = 0; 5 coefficients

<img src="./output_images/cal_undistorted_0.png" width="500">



flags = CALIB\_RATIONAL_MODEL; 8 coefficietns

<img src="./output_images/cal_undistorted_1.png" width="500">


flags = CALIB\_RATIONAL\_MODEL | CALIB\_THIN\_PRISM\_MODEL; 12 coefficietns

<img src="./output_images/cal_undistorted_2.png" width="500">



flags = CALIB\_RATIONAL\_MODEL | CALIB\_THIN\_PRISM\_MODEL | CALIB\_TILTED\_MODEL; 14 coefficietns

<img src="./output_images/cal_undistorted_3.png" width="500">

*(To use THIN\_PRISM\_MODEL, you need to update OpenCV to version 3.2)*

We can also compare original and undistorted images by superimposing them.

Here is original and 'rational model' superposition.

<img src="./output_images/dst_und1_stack.png" width="500">

You can see that all differenes are near image borders, at the center both images 
coinside. And this is a normal situation for almost any lens. I have only one lens
in my collection, which may give signigicant distortions at the center of
the frame - at that one is the 'artistic' LensBaby 2.0.

<img src="./output_images/lensbaby.jpg" width="300">

### Pipeline (single images)

Examples above show that lens distortion is minimal at the center of the images,
where we expect lines to appear. So, I'll not publish 'undistorted lanes' here - 
you'll see effect of undistortion few linew below.



#### 2. Colors, gradients, thresholds.



At the begining, I used a cobination of:
- color threshold in S-channels of the HLS color space
- sobel magntiude threshold
- sobelX treshold

The code for thresholding (adoprted from the lesson with
minor modifications) you can find in the file `thresholds.py` 
(lines 25 - 144).

In this image you see 'test2' image with overlayed binary images: 
- blue for S-channel threshold
- green for sobel magnitude
- red for sobelX

<img src="./output_images/test2+thr.png" width="800">

Here you can clearly see, pronounced undistortion effect on the sign - but
not on the lanes.

Yet, after several tests (inluding 'challenge' video and several far more challenging
videos from my huge collection), I reverted to the simple pipeline I used in P1:
thresholding of the difference between red channel of the image and it's blurred version:
```
    red_channel = image[:,:,0];
    blurred = cv2.medianBlur(red_channel,25);
    diff = cv2.subtract(red_channel, blurred)
    binary = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)[1];
    out = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
```
(file `thresholds.py`, lines 14-24).

It works better in many different situations - and it is four times faster!

Result of the thresholding is shown below:

<img src="./output_images/im4_2(thr).png" width="600">



#### 3. Perspective transform.

To calculate perspective transform, I used provided file `straight_lines1.jpg'.
The image was undistorted using intrinsic camera parameters calculated in the
fiers step. Then I found line coordinates with the help of detectLanes() function
developed in [P1](https://github.com/diz-vara/CarND-P1/blob/master/P1.ipynb).

<img src="./output_images/straight0+lines.png" width="600">

Two original lines coordinates were calculated by `detectLanes()`:
```
src = np.float32(
   [[ lines[0][0], lines[0][1]],
    [ lines[0][2], lines[0][3]], 
    [ lines[1][0], lines[1][1]],
    [ lines[1][2], lines[1][3]], 
```
which approximately gives:

```
src = np.float32(
   [[ 268, 676],
    [ 580, 460], 
    [1046, 676], 
    [697,  460]])
``` 
 To calculate transformation that will make these lines vertical 
 (and full-height), destination points have same x-coordinates,
 and y-coodrdinates corresponding to the size of the image: 
```
dst = np.float32(
   [[ lines[0][0], height],
    [ lines[0][0], 0], 
    [ lines[1][0], height], 
    [ lines[1][0],  0]])
 ```
where height = image.shape[0]

Result of coordinate transformation is shown below: 
 
<img src="./output_images/warped.png" width="600">

(In the actual pipeline, I applyed coordinate transformation **after** thresholding - 
this sequence gives more consistent results.)

#### 4. Lane-line pixels and polynomial fit

At this point, it became possible to apply thresholding, undistortion and perspective
transformation - and obtain bird-eye view of the curved lines:

I also removed noise (small particles) by simple 'open' operation
(`cv2.MORPH_OPEN`)

<img src="./output_images/lines_warped.png" width="400">

Initially, lines were identified by a sliding window search (file `window_fit.py`, lines 13-74).
I replaces explicit points selection from the lession by OpenCV `findNonZero()`, apllyed to
the search rectangles.
 
Process of the search is illustrated below:

<img src="./output_images/sliding_search.png" width="400">

In subsequent frames, I just masked binary image with the mask, defined by previous
polinomial fit:

<img src="./output_images/mask.png" width="400">

(I used separate values for left-line and right-line masks).

When pixels belonging to the left and right lines were identified, I used 
`np.polyfit()` to compute second-order polynomial. Computed polynomials were
exponentioally smoothed (file `line.py`, lines 42-67).

After computation, the resulted region was plotted on the separate 'overaly' image
(file `window_fit.py`, lines 137-144), 

<img src="./output_images/im4_4(lane).png" width="400">


Inverted percpective transformation matrix was applyed to the overlay:

<img src="./output_images/im4_5(lane_unw).png" width="400">

and mask image was formed
(file `window_fit.py`, lines 147-160)


Actual lane curvature was calculated by a given formula and apporximate calibartion
values (file `line.py`, lines 68-70), resulted values were averaged, and 
shift from the center was estimated (file `window_fit.py`, lines 162-167)

<img src="./output_images/im4_6(result).png" width="400">

Then this pipeline was applyed to the projet video:
(file `P4.py`, lines 67-81)


Here's a [link to my video result](./project_result.mp4) - 
and here's a [result of the 'challenge_video' processing](./challenge_result.mp4)

---

### Discussion

Considering all things we learned in this module, I still think that it is just a toy,
student example, first glimpse of the real problem.

* Line extraction methods (my 'simple' and sofisticated thresholds from the lessons) will not work in
*all* possible light conditions. May be, it is just a question of parameters - but then we should have
some additional module that will analyse the picture (sequence of pictures) - and adjust parameters.
* This approach will not work in a common situation of a lane crossing: if the relative motion is fast,
we'll (probably) loose the track of the lines, and histogram analysis will not help us to separate them,
because line will cross the center of the frame.

    <img src="./output_images/crossing.png" width="400">
    
    Other line separation methods can help - I've succesfully apllyed 
    `AgglomerativeClustering` from the `sklearn` package - but anyway we'll 
    need some priors

* Line detection can easily be distracted by nearby cars, additonal marks and writings on the
road. Add to it tram lines, buildings, advertisments and pedestrians - and you'll see that lane
navigation in the city is a really hard problem.

One can tell that it is possible to build deep network that will take an input image (or video in
the case of RNN) - an return all lanes. But I do not think it is a right answer to the question: not 
only because of the 'black-box'ness' of the deep networks - but because ANNs learn from a lot of 
frequent data and can't answer properly to the situation they never saw.

From my point of view, the better way is to create some flexible and complex (but still manageble) model
of the world (including road configuration, lanes, nearby cars, possible obstacles etc.) - and fit this
model parameters to the observed data. In a way, we did it in this project - but our model was 
very simple (consisting of two parallel lines) - and we fit it in a straightforward, non-probabilistic
way.

Thinking of consiense solutions, I try to keep track on Bayesians - groups of [Zoubin Ghahramani](http://mlg.eng.cam.ac.uk/zoubin/)
and [Raquel Urtasun](http://www.cs.toronto.edu/~urtasun/). And, as [yesterday news](http://tcrn.ch/2pSyzMN) showed
that now they'll work together - we know where to look for such models.

 

 