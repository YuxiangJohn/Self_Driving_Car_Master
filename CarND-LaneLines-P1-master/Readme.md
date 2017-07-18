# **Finding Lane Lines on the Road** 




---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Process the image and the video to draw lane lines



[//]: # (Image References)

[image1]: ./test_images_output/result_solidYellowLeft.jpg

---

### Description

### 1. Pipline

My pipeline consisted of 5 steps. 
* grayscale: **cv2.cvtColor** Convert the images to grayscale.
* blur: **cv2.GaussianBlur** Blur the grayscale images using the Gaussian blur.
* canny: **cv2.Canny** Use Canny edges detection to find egdes on the blur images.
* mask: Restrict the part of the image so that the not intersting part would be eliminated.
* hough: **cv2.HoughLinesP** Use the Hough transformation to find the lines on the masked image.
* draw: In order to draw a solid and fix line on the left and right lanes, I use the lines found in the hough transformation, find the slope of the lines and distinguish the left and right by the postive and negative of slope, get two points per line to make a point set. Then I use Least squares method to find a line which fits most of these points(**numpy.polyfit**).
* weighted: Merges the drawed lines and the original image.



![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
