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
        self.curvatures = []
        self.distances_from_center = []
        self.input_video_clip = VideoFileClip(video_path)
        self.output_video_clip = self.input_video_clip.fl(self.process_frame)

    def process_video(self):
        self.output_video_clip.write_videofile(self.output_path, audio=False)

    def draw_text(self, frame, text, x, y):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)

    def process_frame(self, gf, t):
        image = gf(t)

        undistorted = cv2.undistort(image, self.calibration.camera_matrix, self.calibration.distortion_coefficients, None)
        filtered = get_edge_mask(undistorted)
        warped = warp_birds_eye(filtered, self.source_points, self.destination_points)
        windows, left_window_points, right_window_points = sliding_window(warped)
        left_line = LaneLine(warped, left_window_points)
        right_line = LaneLine(warped, right_window_points)

        self.left_lines.append(left_line)
        self.right_lines.append(right_line)

        left_line.fit()
        right_line.fit()

        filtered = get_edge_mask(undistorted, input_image_color_space='rgb', return_all_channels=True)
        filtered[filtered == 1] = 255
        original_birds_eye = warp_birds_eye(undistorted, self.source_points, self.destination_points)
        filtered_birds_eye = warp_birds_eye(filtered, self.source_points, self.destination_points)

        y = left_line.generate_y()

        lane_drawing = np.zeros_like(original_birds_eye)
        left_x = np.median(np.array([ l.evaluate() for l in self.left_lines[-8:] ]), axis=0)
        left_points = np.vstack([left_x, y]).T

        right_x = np.median(np.array([ l.evaluate() for l in self.right_lines[-8:] ]), axis=0)
        right_points = np.vstack([right_x, y]).T

        all_points = np.concatenate([left_points, right_points[::-1], left_points[:1]])

        cv2.fillConvexPoly(lane_drawing, np.int32([all_points]), (0, 255, 0))

        unwarped_lane_drawing = warp_birds_eye(lane_drawing, self.source_points, self.destination_points, reverse=True)
        original_perspective_windows = warp_birds_eye(windows, self.source_points, self.destination_points, reverse=True)

        frame = cv2.addWeighted(undistorted, 1.0, unwarped_lane_drawing, 0.2, 0)

        l = np.average(np.array([line.camera_distance() for line in self.left_lines[-8:]]))
        r = np.average(np.array([line.camera_distance() for line in self.right_lines[-8:]]))
        if l - r > 0:
            self.draw_text(frame, '{:.3} cm right of center'.format((l - r) * 100), 20, 115)
        else:
            self.draw_text(frame, '{:.3} cm left of center'.format((r - l) * 100), 20, 115)

        self.curvatures.append(np.mean([left_line.curvature_radius(), right_line.curvature_radius()]))
        curvature = np.average(self.curvatures[-8:])
        self.draw_text(frame, 'Radius of curvature:  {:.3} km'.format(curvature / 1000), 20, 80)

        return frame