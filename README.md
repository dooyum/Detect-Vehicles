## Project Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Estimate a bounding box for vehicles detected.
* Create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Run the car detection pipeline on a video stream.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/hog1.png
[image3]: ./output_images/hog2.png
[image4]: ./output_images/hog3.png
[image5]: ./output_images/hog4.png
[image6]: ./output_images/hog5.png
[image7]: ./output_images/hog6.png
[image8]: ./output_images/bboxes.png
[image9]: ./output_images/bboxes2.png
[image10]: ./output_images/bboxes3.png
[image11]: ./output_images/bboxes4.png
[image12]: ./output_images/bboxes5.png
[image13]: ./output_images/bboxes6.png
[image14]: ./output_images/bboxes7.png
[image15]: ./output_images/bboxes8.png
[image16]: ./output_images/bboxes9.png
[image17]: ./output_images/bboxes10.png
[image18]: ./output_images/bboxes11.png
[image19]: ./output_images/bboxes12.png
[image20]: ./output_images/bboxes13.png
[image21]: ./output_images/heatmap1.png
[image22]: ./output_images/heatmap2.png
[image23]: ./output_images/heatmap3.png
[image24]: ./output_images/heatmap4.png
[image25]: ./output_images/heatmap5.png
[image26]: ./output_images/heatmap6.png
[image27]: ./output_images/heatmap7.png
[image28]: ./output_images/heatmap8.png
[image29]: ./output_images/heatmap9.png
[image30]: ./output_images/heatmap10.png
[image31]: ./output_images/heatmap11.png
[image32]: ./output_images/detection_example.png
[video1]: ./output_videos/project_video.mp4


### Read training images

In the first code cell of the IPython [notebook](./notebook.ipynb) I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]


### Extract image features

Next I extracted feature vectors from the training images in order to train a `car`/`not car` classifier.

#### 1. Extraction of Color features from the training images.

The first two feature vectors extracted from the image were color features. This feature exrtaction can be seen in cells 7 through 9 of the [notebook](./notebook.ipynb). 

The first of the color features was created by bining the image using an `8x8 spatial size`.

The second was derived by creating a histogram for each channel in the image on combining them into a single vector.

Training a classifier with these color features combined gave me an accuracy of 97.5%.

#### 2. Extraction of HOG features from the training images.

Next, I applied a HOG transform to the training images in order to use the shape of the vehicles as a feature vector. This was implemented in cells 11 through 13 of the [notebook](./notebook.ipynb).

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from the training data and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the multiple color spaces and HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]

##### Choice of HOG parameters.

I tried various combinations of HOG parameters and color channels and observed the following:

- For Color Channels: Using all YUV channels worked best across all training images
    - RGB is best on very clear images with bright(white) cars where all channel values are high, but fails when shadows are present.
    - LUV, YUV and YCrCb higlight very similar features
    - HLS and HSV are prone to be noisy
    - The L and U channels of HLS, LUV, YUV and YCrCb adjust well to color changes
    - The V and BC channels of LUV, YUV and YCrBc higlight the tail and head lights most effectively
    - Combining all channels of LUV will work best
- For HOG params:
    - 8 orientations would work okay (north, north-east, east, south-east, south, south-west, west, north-west),
      but 12 orientations had higher precision, before hitting diminishing returns on training speed.
      So 12 worked best.
    - Less pixels per cell give higher definition but is slower. 8 pixels per cell worked best.

#### 3. Train a classifier

I combined the Color and HOG feature vectors and used them to train a linear SVM classifier. This can be seen in cells 13 through 17 of the [notebook](./notebook.ipynb).

Using the combined feature vectors improved the trained classifier and I got 100% accuracy with its test predictions.

### Sliding Window Search

The next step in my implementation was to perform a sliding window search of the image and detect vehicles.
This involved croping the image to the size of multiple bounding boxes and searching within each bounding box for a vehicle.
The sliding window search was implemented in cells 18 through 30 of the [notebook](./notebook.ipynb).

Here's an example of an image with multiple bounding boxes to be searched.
![alt text][image8]

#### 1. Window bounds
Since we do not have any expectations that a vehicles will be in the sky(top half of the car camera image), we could resonably apply the window search to the lower half of the photo.
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]

#### 2. Bounding box scaling
Vehicles tend to vary in size and where they appear in relation to the perpective of the cars camera could determine its size in the image. As a result, a single bounding box size will not be sufficient to detect all vehicles.
To resolve this issue, I created bounding boxes of varying scales, and applied them to bounds within the image where a vehicle of that size could reasonably occur.
For example smaller bounding boxes tend to occur closer to the horizon while bigger bounding boxes will be expected closeer to the car camera.
I came up with the following formula for computing the reasonable bounds based on the bounding box scale: 
`y_boundary_end = y_boundary_start + (base_window_height * scale * 1.5)`.

This led to more precise vehicle detections at all perspectives of the car camera image.
![alt text][image15]
![alt text][image16]
![alt text][image17]
![alt text][image18]
![alt text][image19]
![alt text][image20]


### Eliminating outliers

#### 1. Data heat map

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap of recurring pixels and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected. This was all performed in cells 32 through 34. 

Here's an example result showing the heatmap from a series of test images, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the each image:
![alt text][image21]
![alt text][image22]
![alt text][image23]
![alt text][image24]
![alt text][image25]


#### 2. Historical vehicle detection
In order to smoothen the detection of vehicles from frame to frame of a video stream and ensure outliers don't blink in and out, I created a class to hold information about the vehicle detection of the current frame. The bounding boxes that had been detected in the current frame are saved along with information about the `n` most recent frames detected. A second heat map is then created from all the bounding boxes of the last `n` frames and a second stricter threshold is applied. Only pixels that are prominent across all `n` frames are used to create a final bounding box of a high confidence vehicle detection. This was all implemented from cells 35 through 38.
The following is an example of a heat map created based of historical data of the 2 most recent frames:
![alt text][image26]
![alt text][image27]
![alt text][image28]
![alt text][image29]
![alt text][image30]
![alt text][image31]


### Vehicle detection pipeline

Ultimately I searched for vehicles using four bounding box scales (1.0,1.5,2.0,2.5), all 3 channels of the YUV color space, HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here's a snapshot of the final output video:

![alt text][image32]
---

### Final vehicles detection video
Here's a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

#### 1. What worked well?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

I took an approach that valued features from images which provided information on the shape and color of the objects in the image, rather than just the color. Training a classifier using color features alone would have resulted in more false positives as items with vibrant colors like road signs could be missclassified.

The attention to the fact that the varying sizes of vehicles in an image should be considered, resulted in much tighter bounding boxes and better all round detection.

#### 2. What could be improved?

Although the pipeline works relatively well for detecting vehicles in the accompanying video, it may not perform as well in areas that are heavily shaded. The pipeline could be improved by providing more labeled training examples with such data.

Right now the pipeline might be a little too strict or too liberal for any given threshold especially the heatmap hot pixel threshold. As an improvement, the threshold could be a factor of the total number of hot pixels available in each frame e.g. retain all pixels above the 25th percentile.

Although the `LinearSVC()` classifier worked well, a better classifier could be built by tweaking parameters manually to find the best performing one.

The use of historical detection information to validate new detections and rule out outliers was also a useful approach.