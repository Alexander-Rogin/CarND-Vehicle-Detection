# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/sliding_window.png
[image5]: ./output_images/bboxes_heat_labels.png
[image6]: ./output_images/output_bboxes.png
[video1]: ./output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 100 through 112 of the file called `features.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found that 6 HOG orientations, (8, 8) pixels per cell and (2, 2) cells per block gave quite clear distinction between car and noncar images.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using SciKit Learn LinearSVC with default parameters. The training set consisted of KITTI and GTI imagess. I used 20% of the training set for validation. At the end of tuning feature parameters, this SVM was able to classify test images with the accuracy around 99%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search for vehicles in the lower part of the image - between 400 and 640 pixels in the `y` coordinates. Experiments showed that scales (1, 1.5, 2, 2.5) with overlapping of 0.66 in `x` and `y` gave me the best detection.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.

For processing videos, I used detections from previous frames to filter false-positive detections and decrease probability of misses.

Here's an example result showing the heatmap from a set of the video frames, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][image5]

Here the resulting bounding boxes are drawn onto a frame:
![alt text][image6]


---

### Discussion

#### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

After brief overview of the training data, it seemed to me that there are no truck images. This will likely lead to failure in detecting trucks. Also, trucks are much larger than cars, especially sideways. My sliding window and its scales are tuned to detect cars and may perform really poorly with larger vehicles. Additional training on trucks and further experiments with the detection scales might help.

Another possible issue that might turn up is different weather conditions, such as heavy rains, blizzards, fog, as well as poor ligthning. Here again additional training and further experiments with color spaces are required.

Last but not least, processing time is really long and the current implementation is far from being realtime. More research should be done in this area to make the pipeline faster.