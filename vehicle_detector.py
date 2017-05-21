import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import sys

import sliding_window as sw
import model
import features as feat

def getRoiParams(img, xy_window):
    x_start_stop = [0, img.shape[1]]
    y_start_stop = [400, 640]

    # if xy_window[0] == xy_window[1]:
    #     center = int(img.shape[1] / 2)
    #     x_start_stop[0] = center - 4 * xy_window[0]
    #     x_start_stop[1] = center + 4 * xy_window[0]
    # y_start_stop[1] = (y_start_stop[0] + 2 * xy_window[1])

    if x_start_stop[0] < 0:
        x_start_stop[0] = 0
    if x_start_stop[1] >= img.shape[1]:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] < 0:
        y_start_stop[0] = 0
    if y_start_stop[1] >= 640:
        y_start_stop[1] = 640
    return x_start_stop, y_start_stop

def getWindows(image):
    xy_windows = [
        (64, 64),
        (128, 64),
        (128, 128),
        (192, 128),
        (192, 192),
        (256, 192),
        (256, 256),
        (320, 256),
        (320, 320),
        (320, 192),
        (384, 192)]

    ret_windows = []
    for xy_window in xy_windows:
        x_start_stop, y_start_stop = getRoiParams(image, xy_window)

        windows = sw.slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop, 
                        xy_window=xy_window, xy_overlap=(0.75, 0.75))

        ret_windows += windows
    return ret_windows


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        # print(window)
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = feat.single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


def getClassifier(train=False, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    if train:
        return model.train(color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    return model.load()

from scipy.ndimage.measurements import label
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

class VehicleDetector:
    def __init__(self, train):
        ### TODO: Tweak these parameters and see how the results change.
        # Color:
        self.hist_feat = True # Histogram features on or off
        self.color_space = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.hist_bins = 32    # Number of histogram bins (16)

        # HOG
        self.hog_feat = True # HOG features on or off
        self.orient = 6  # HOG orientations (9)
        self.pix_per_cell = 8 # HOG pixels per cell
        self.cell_per_block = 2 # HOG cells per block
        self.hog_channel = 1 # Can be 0, 1, 2, or "ALL"

        # Spatial
        self.spatial_feat = True # Spatial features on or off
        self.spatial_size = (32, 32) # Spatial binning dimensions (16, 16)

        if train:
            self.X_scaler, self.svc = model.train(color_space=self.color_space, spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
                        orient=self.orient, pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block, 
                        hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, hist_feat=self.hist_feat, hog_feat=self.hog_feat)
        else:
            self.X_scaler, self.svc = model.load()

    def processImage(self, image, pngJpg=True):
        draw_image = np.copy(image)
        if pngJpg:
            image = image.astype(np.float32)/255
        windows = getWindows(image)
        hot_windows = search_windows(image, windows, self.svc, self.X_scaler, color_space=self.color_space, 
                            spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
                            orient=self.orient, pix_per_cell=self.pix_per_cell, 
                            cell_per_block=self.cell_per_block, 
                            hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
                            hist_feat=self.hist_feat, hog_feat=self.hog_feat)

        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)


        heat = np.zeros_like(draw_image[:,:,0]).astype(np.float)
        # Add heat to each box in box list
        heat = add_heat(heat, hot_windows)
    
        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 2)

        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(draw_image), labels)

        # fig = plt.figure()
        # plt.subplot(121)
        # plt.imshow(draw_img)
        # plt.title('Car Positions')
        # plt.subplot(122)
        # plt.imshow(heatmap, cmap='hot')
        # plt.title('Heat Map')
        # fig.tight_layout()

        # return window_img


        
        return draw_img


        



image_path = 'test_images/test1.jpg'
train = False
useVideo = False

argc = len(sys.argv)
for i in range(argc):
    if sys.argv[i] == '--image' and i < argc - 1:
        i += 1
        image_path = sys.argv[i]
    elif sys.argv[i] == '--train':
        train = True
    elif sys.argv[i] == '--video' and i < argc - 1:
        i += 1
        video_path = sys.argv[i]
        useVideo = True


vd = VehicleDetector(train)

if not useVideo:
    image = mpimg.imread(image_path)
    
    result = vd.processImage(image)

    plt.imshow(result)
    plt.show()
else:
    from moviepy.editor import VideoFileClip
    video = VideoFileClip(video_path)
    processed_video = video.fl_image(vd.processImage)
    processed_video.write_videofile('output.mp4', audio=False)