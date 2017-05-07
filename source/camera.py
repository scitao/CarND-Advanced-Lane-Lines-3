import cv2
import glob
import numpy as np
import pickle

class Calibration(object):

    def __init__(self, images_glob, pattern_size, calibration_path, output_status=False):
        self.images_glob = images_glob
        self.pattern_size = pattern_size
        self.camera_matrix = None
        self.distortion_coefficients = None
        self.output_status = output_status
        self.calibration_path = calibration_path

    def calibrate(self):
        '''
        Loads a cached calibration or performs a new calibration and caches it to disk.
        '''
        try:
            self.camera_matrix, self.distortion_coefficients = self.load_calibration()
            if self.output_status is True:
                print('Loaded saved calibration file')
        except FileNotFoundError:
            if self.output_status is True:
                print('No saved calibration file found, re-calibrating...')
            _, self.camera_matrix, self.distortion_coefficients, _, _ = self.do_calibration()
            if self.output_status is True:
                print('Regenerated and saved calibration')

    def load_calibration(self):
        '''
        Loads a saved calibration
        '''
        contents = pickle.load(open(self.calibration_path, 'rb'))
        return contents['camera_matrix'], contents['distortion_coefficients']
        
    def do_calibration(self, draw_corners_path=None, save_calibration_filename=None):
        '''
        Return calibration values for the images found in images_glob
        '''
        object_points = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        object_points[:,:2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1,2)

        all_object_points = []
        all_image_points = []
        img_size = None

        for index, filename in enumerate(glob.glob(self.images_glob)):
            image = cv2.imread(filename)
            
            if img_size is None:
                img_size = (image.shape[1], image.shape[0])
            if img_size[0] != image.shape[1] or img_size[1] != image.shape[0]:
                raise Exception('Image size should be the same for all images')

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            ret, found_corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
            if ret == False:
                continue

            all_object_points.append(object_points)
            all_image_points.append(found_corners)

            if draw_corners_path is not None:
                cv2.drawChessboardCorners(image, self.pattern_size, found_corners, ret)
                write_name = draw_corners_path + '/corners_found' + str(index + 1) + '.jpg'
                cv2.imwrite(write_name, image)
                
        _, camera_matrix, distortion_coefficients, _, _ = cv2.calibrateCamera(
            all_object_points, 
            all_image_points, 
            img_size, 
            None, 
            None
        )    
        
        contents = {}
        contents['camera_matrix'] = camera_matrix
        contents['distortion_coefficients'] = distortion_coefficients
        pickle.dump(contents, open(self.calibration_path, 'wb'))
        return _, camera_matrix, distortion_coefficients, _, _

def warp_birds_eye(image, source_points, destination_points, reverse=False):
    '''
    Shift perspective in image from looking down the road to a bird's-eye
    view above the camera looking straight down.
    '''

    def shift_perspective(image, source_points, destination_points):
        image_size = (image.shape[1], image.shape[0])
        matrix = cv2.getPerspectiveTransform(source_points, destination_points)
        warped = cv2.warpPerspective(image, matrix, image_size)
        return warped

    (h, w) = (image.shape[0], image.shape[1])
    if reverse is True:
        return shift_perspective(image, destination_points, source_points)
    else:
        return shift_perspective(image, source_points, destination_points)

def corners_unwarp(img, nx, ny, mtx, dist):
    '''
    Undistort and shift perspective of camera calibration images.
    '''
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == False:
        return None, None
    
    # If we found corners, draw them! (just for fun)
    cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
    # Choose offset from image corners to plot detected corners
    # This should be chosen to present the result at the proper aspect ratio
    # My choice of 100 pixels is not exact, but close enough for our purpose here
    offset = 100 # offset for dst points
    # Grab the image shape
    img_size = (gray.shape[1], gray.shape[0])

    # For source points I'm grabbing the outer four detected corners
    src = np.float32([
        corners[0], 
        corners[nx - 1], 
        corners[-1], 
        corners[-nx]
    ])
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    dst = np.float32([
        [offset, offset], 
        [img_size[0] - offset, offset], 
        [img_size[0] - offset, img_size[1] - offset], 
        [offset, img_size[1] - offset]
    ])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(undist, M, img_size)

    # Return the resulting image and matrix
    return warped, M