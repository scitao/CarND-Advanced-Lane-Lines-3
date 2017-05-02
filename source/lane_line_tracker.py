import cv2
from PIL import Image
from moviepy.editor import VideoFileClip
import numpy as np
from lane_line import LaneLine

from camera import warp_birds_eye
from thresholding import get_edge_mask
from window import sliding_window

class LaneLineTracker(object):

    def __init__(self, video_path, calibration, source_points, destination_points, output_path='../output.mp4'):
        self.output_path = output_path
        self.source_points = source_points
        self.destination_points = destination_points
        self.calibration = calibration
        self.left_lines = []
        self.right_lines = []
        self.input_video_clip = VideoFileClip(video_path)
        self.output_video_clip = self.input_video_clip.fl(self.process_frame)

    def process_video(self):
        self.output_video_clip.write_videofile(self.output_path, audio=False)

    def process_frame(self, gf, t):
        image = gf(t)

        undistorted = cv2.undistort(image, self.calibration.camera_matrix, self.calibration.distortion_coefficients, None)
        filtered = get_edge_mask(undistorted)
        warped = warp_birds_eye(filtered, self.source_points, self.destination_points)
        windows, left_window_points, right_window_points = sliding_window(warped)
        left_line = LaneLine(warped, left_window_points)
        right_line = LaneLine(warped, right_window_points)

        left_line.fit()
        right_line.fit()

        filtered = get_edge_mask(undistorted, input_image_color_space='rgb', return_all_channels=True)
        filtered[filtered == 1] = 255
        original_birds_eye = warp_birds_eye(undistorted, self.source_points, self.destination_points)
        filtered_birds_eye = warp_birds_eye(filtered, self.source_points, self.destination_points)

        y = np.linspace(0, 719, num=720)

        left_x = left_line.evaluate(y)
        left_points = np.vstack([left_x, y]).T
        cv2.polylines(windows, np.int32([left_points]), False, (255, 255, 255), 3)

        right_x = right_line.evaluate(y)
        right_points = np.vstack([right_x, y]).T
        cv2.polylines(windows, np.int32([right_points]), False, (255, 255, 255), 3)


        original_perspective_windows = warp_birds_eye(windows, self.source_points, self.destination_points, reverse=True)



        return self.merge_images(
            image,
            original_perspective_windows,
            filtered_birds_eye,
            filtered
        )

    def merge_images(self, np_image1, np_image2, np_image3, np_image4):
        image1 = Image.fromarray(np.uint8(np_image1))
        image2 = Image.fromarray(np.uint8(np_image2))
        image3 = Image.fromarray(np.uint8(np_image3))
        image4 = Image.fromarray(np.uint8(np_image4))

        (width1, height1) = image1.size
        (width2, height2) = image2.size
        (width3, height3) = image3.size
        (width4, height4) = image4.size

        result_width = width1 + width2
        result_height = height1 + height3

        result = Image.new('RGB', (result_width, result_height))
        result.paste(im=image1, box=(0, 0))
        result.paste(im=image2, box=(width1, 0))
        result.paste(im=image3, box=(0, height1))
        result.paste(im=image4, box=(width1, height1))

        return np.array(result)