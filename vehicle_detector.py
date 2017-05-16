import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

import sliding_window as sw
import hog

def getRoiParams(img, xy_window):
    x_start_stop = [0, img.shape[1]]
    y_start_stop = [400, 680]

    if xy_window[0] == xy_window[1]:
        center = int(img.shape[1] / 2)
        x_start_stop[0] = center - 3 * xy_window[0]
        x_start_stop[1] = center + 3 * xy_window[0]
    y_start_stop[1] = (y_start_stop[0] + int(1.5 * xy_window[1]) + 1)

    if x_start_stop[0] < 0:
        x_start_stop[0] = 0
    if x_start_stop[1] >= img.shape[1]:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] < 0:
        y_start_stop[0] = 0
    if y_start_stop[1] >= img.shape[0]:
        y_start_stop[1] = img.shape[0]
    return x_start_stop, y_start_stop

def getWindows(image):
    xy_windows = [
        (64, 64),
        (128, 64),
        (128, 128),
        (192, 128),
        (192, 192),
        (256, 192),
        (320, 192),
        (384, 192)]

    ret_windows = []
    for xy_window in xy_windows:
        x_start_stop, y_start_stop = getRoiParams(image, xy_window)

        windows = sw.slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop, 
                        xy_window=xy_window, xy_overlap=(0.5, 0.5))

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
                    orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        # features = single_img_features(test_img, color_space=color_space, 
        #                     spatial_size=spatial_size, hist_bins=hist_bins, 
        #                     orient=orient, pix_per_cell=pix_per_cell, 
        #                     cell_per_block=cell_per_block, 
        #                     hog_channel=hog_channel, spatial_feat=spatial_feat, 
        #                     hist_feat=hist_feat, hog_feat=hog_feat)
        features = hog.extract_single_image_features(test_img, cspace=color_space, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


def getClassifier(train=False):
    if train:
        return hog.train()
    return hog.loadModel()



image = mpimg.imread('test_images/test6.jpg')
X_scaler, svc = getClassifier()
windows = getWindows(image)

draw_image = np.copy(image)


### TODO: Tweak these parameters and see how the results change.
color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)                       

window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    

plt.imshow(window_img)
plt.show()






# print(len(windows))

# for xy_window in xy_windows:
#     x_start_stop, y_start_stop = getRoiParams(image, xy_window)

#     windows = sw.slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop, 
#                         xy_window=xy_window, xy_overlap=(0.5, 0.5))

#     window_img = sw.draw_boxes(image, windows, color=(0, 0, 255), thick=6)
#     plt.imshow(window_img)
#     plt.show()