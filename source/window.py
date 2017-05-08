import cv2
import numpy as np

WINDOW_WIDTH = 100
WINDOW_HEIGHT = 80
MARGIN = 50

def draw_windows(image, window_centroids, window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT, margin=MARGIN):
    image = cv2.cvtColor(image.astype(np.uint16), cv2.COLOR_GRAY2RGB)
    h = image.shape[0]
    w = image.shape[1]
    for level, window in enumerate(window_centroids):
        for side in [0, 1]:
            y_min = int(h - (level + 1) * window_height) 
            y_max = int(h - level * window_height)
            x_min = max(0, int(window[side] - window_width / 2))
            x_max = min(int(window[side] + window_width / 2), w)
            cv2.rectangle(
                image, 
                (x_min, y_min), 
                (x_max, y_max), 
                color=(0, 1, 0), thickness=4
            )
    return image

def find_window_centroids(warped, window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT, margin=MARGIN):
    window_centroids = [] # Store the (left,right) window centroid positions per level
    conv_signals = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
    l_conv_signal = np.convolve(window, l_sum)
    l_center = np.argmax(l_conv_signal) - window_width/2
    r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
    r_conv_signal = np.convolve(window, r_sum)
    r_center = np.argmax(r_conv_signal) - window_width / 2 + int(warped.shape[1] / 2)

    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    conv_signals.append((
        np.max(l_conv_signal),
        np.max(r_conv_signal),
    ))

    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(warped.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        l_max = np.max(conv_signal[l_min_index:l_max_index])
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        r_max = np.max(conv_signal[r_min_index:r_max_index])
        # Add what we found for that layer
        conv_signals.append((l_max, r_max))
        
        if conv_signals[-1][0] < 0.5 * conv_signals[-2][0]:
            l_center = l_min_index
        if conv_signals[-1][1] < 0.5 * conv_signals[-2][1]:
            r_center = r_min_index

        window_centroids.append((l_center,r_center))

    return window_centroids

def sliding_window(image, window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT, margin=MARGIN):

    def window_mask(window_width, window_height, img_ref, center,level):
        output = np.zeros_like(img_ref)
        h = img_ref.shape[0]
        w = img_ref.shape[1]
        y_min = int(h - (level + 1) * window_height) 
        y_max = int(h - level * window_height)
        x_min = max(0, int(center - window_width / 2))
        x_max = min(int(center + window_width / 2), w)
        output[y_min:y_max, x_min:x_max] = 1
        return output

    warped = np.copy(image)

    window_centroids = find_window_centroids(warped, window_width, window_height, margin)

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

    return output, l_points, r_points