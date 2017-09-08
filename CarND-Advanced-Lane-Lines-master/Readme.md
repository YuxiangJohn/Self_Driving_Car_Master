# Advanced Lane Finding



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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

### Code:
This project requires python 3.5 and the following dependecies:
- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [OpenCV](http://opencv.org/)
- [MoviePy](http://zulko.github.io/moviepy/)

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "P4.ipynb" .

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![](output_images/1.png)

### Pipeline (single images)

#### 1. Distortion correction

Just like the undistortion for the chessboard, we applied the `undistort()` function to a test image on the road and got the result:

![](output_images/2.png)



#### 2. Perspective transform 

The code for my perspective transform includes a function called `per_transform()`, which appears  in the first code cell under the Perspective transform part on the IPython notebook.  I chose the hardcode the source and destination points(src,dst) in the following manner:

```python
src = np.float32([[220,720],[1110,720],[722,470],[570,470]])
dst = np.float32([[320,720],[920,720],[920,1],[320,1]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 220, 720      | 320, 720      | 
| 1110, 720     | 920, 720      |
| 722, 470      | 920, 1        |
| 570, 470      | 320, 1        |

The applied the transform to the test image and we can see the result:

![](output_images/3.png)
![](output_images/4.png)


#### 2.Color and Gradient threshold.

I used a combination of color and gradient thresholds to generate a binary image. I created a function `threshold()` to do these. In this function, the input image will first undistrot by the `undistort()` function. Then do the perspective transform as above..Apply gradient threshold(min=20,max=100) on the horizontal gradient to the grayscale image. Convert the RGB image to HLS color space and get the s_channel. Apply threshold(min=90,max=255) to the s_channel. Finally combine the s_channel and the gradient to get the final result.
Here's an example of my output for this step. 

![](output_images/5.png)

#### 4. Identify lane-line pixels and fit their positions with a polynomial

The method to detect lane line can sperate into two part. The first is to get the histogram of the image on the horizontal direction. We can see that the Peak in the first half indicates the likely position of the left line; the peak in the second half indicates the likely postion of the right line.

![](output_images/6.png)

The sencond is to apply the sliding window. The window size is 10. The histogram gave us the location of lane lines. We just identify all non zero pixels around histogram peaks using numpy function `numpy.nonzeros()`. Then fit a polynomial to each lane using the numpy fucntion `numpy.polyfit()`. 

![](output_images/8.png)
![](output_images/7.png)

#### 5. Calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The formula to calculate the radius of curvature can been seen here: [Curvature](https://www.intmath.com/applications-differentiation/8-radius-curvature.php)
We calculate the radius of curvature in the bird eye graph first. Then we transform the result back to the real world space.
Then we found the left bottom and right bottom of two lines, average the postion. The middle point of the original image can be used as the reference mid point to calculate the difference of the average position. So the offset of center is measured.
The code of this part are shown in the code cell under the `measuring Curvature` part.

#### 6. Example of Draw lane

I implemented the overall pipeline in the `Draw lane` part. Here is the example:

![](output_images/9.png)
![](output_images/10.png)
![](output_images/11.png)

---

### Pipeline (video)

#### 1. Final video output.  

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

The pipeline work well in the project video. But the pipeline may fail when there are complicated weather conditions. The light will effect the result too. To build a robust pipeline, we need to combine more channel or threshold.

### Reference
https://chatbotslife.com/robust-lane-finding-using-advanced-computer-vision-techniques-46875bb3c8aa#.l2uxq26sn

https://www.researchgate.net/publication/257291768_A_Much_Advanced_and_Efficient_Lane_Detection_Algorithm_for_Intelligent_Highway_Safety

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5017478/
