import numpy as np
import cv2

def gradient_abs_value_mask(image, sobel_kernel=3, axis='x', threshold=(0, 255)):
    # Take the absolute value of derivative in x or y given orient = 'x' or 'y'
    if axis == 'x':
        sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if axis == 'y':
        sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobel = np.uint8(255 * sobel / np.max(sobel))
    # Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    mask = np.zeros_like(sobel)
    # Return this mask as your binary_output image
    mask[(sobel >= threshold[0]) & (sobel <= threshold[1])] = 1
    return mask

def gradient_magnitude_mask(image, sobel_kernel=3, threshold=(0, 255)):
    # Take the gradient in x and y separately
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the magnitude
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    magnitude = (magnitude * 255 / np.max(magnitude)).astype(np.uint8)
    # Create a binary mask where mag thresholds are met
    mask = np.zeros_like(magnitude)
    mask[(magnitude >= threshold[0]) & (magnitude <= threshold[1])] = 1
    # Return this mask as your binary_output image
    return mask

def gradient_direction_mask(image, sobel_kernel=3, threshold=(0, np.pi / 2)):
    # Take the gradient in x and y separately
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the x and y gradients and calculate the direction of the gradient
    direction = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
    # Create a binary mask where direction thresholds are met
    mask = np.zeros_like(direction)
    # Return this mask as your binary_output image
    mask[(direction >= threshold[0]) & (direction <= threshold[1])] = 1
    return mask

def get_edge_mask(
    image,
    input_image_color_space='bgr',
    l_thresh=(230, 255),
    b_thresh=(145, 255),
    sobel_thresh=(20, 100),
    sobel_direction_thresh=(0.7, 1.3),
    return_all_channels=False
    ):
    '''
    Thresholds an image based on Sobel gradient in the x direction, the L-channel
    in an LUV colorspace, and the B-channel in a LAB colorspace. Returns a mask
    with pixel values of 0 or 1 based on accepted thresholded values.
    '''
    (h, w) = (image.shape[0], image.shape[1])
    image = np.copy(image)
    
    if input_image_color_space == 'bgr':
        luv = cv2.cvtColor(image, cv2.COLOR_BGR2Luv).astype(np.float)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab).astype(np.float)
    elif input_image_color_space == 'rgb':
        luv = cv2.cvtColor(image, cv2.COLOR_RGB2Luv).astype(np.float)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab).astype(np.float)

    l_channel = luv[:,:,0]
    b_channel = lab[:,:,2]
    
    gradient_l_x = gradient_abs_value_mask(l_channel, axis='x', sobel_kernel=3, threshold=sobel_thresh)
    gradient_l_y = gradient_abs_value_mask(l_channel, axis='y', sobel_kernel=3, threshold=sobel_thresh)
    magnitude_l = gradient_magnitude_mask(l_channel, sobel_kernel=3, threshold=sobel_thresh)
    direction_l = gradient_direction_mask(l_channel, sobel_kernel=3, threshold=sobel_direction_thresh)

    gradient_b_x = gradient_abs_value_mask(b_channel, axis='x', sobel_kernel=3, threshold=sobel_thresh)
    gradient_b_y = gradient_abs_value_mask(b_channel, axis='y', sobel_kernel=3, threshold=sobel_thresh)
    magnitude_b = gradient_magnitude_mask(b_channel, sobel_kernel=3, threshold=sobel_thresh)
    direction_b = gradient_direction_mask(b_channel, sobel_kernel=3, threshold=sobel_direction_thresh)

    gradient_mask = np.zeros_like(l_channel)
    gradient_mask[
        ((gradient_l_x == 1) & (gradient_l_y == 1)) |
        ((magnitude_l == 1) & (direction_l == 1)) |
        ((gradient_b_x == 1) & (gradient_b_y == 1)) |
        ((magnitude_b == 1) & (direction_b == 1))
    ] = 1

    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh[0]) & (b_channel <= b_thresh[1])] = 1

    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

    color_binary = np.dstack(( l_binary, gradient_mask, b_binary))
    if return_all_channels is True:
        return color_binary

    mask = np.zeros_like(b_binary)
    mask[(l_binary >= 1) | (gradient_mask >= 1) | (b_binary >= 1)] = 1

    return mask[:,:,None]