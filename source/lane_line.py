import numpy as np

class LaneLine(object):

    def __init__(self, image, window_points):
        self.w = image.shape[1]
        self.h = image.shape[0]
        self.points = np.zeros_like(image)
        self.points[(window_points == 255) & (image == 1)] = 1
        self.x = np.where(self.points == 1)[1]
        self.y = np.where(self.points == 1)[0]
        self.fit_parameters = None

    def fit(self):
        self.fit_parameters = np.polyfit(self.y, self.x, deg=2)
        return self.fit_parameters

    def generate_y(self):
        return np.linspace(0, self.h - 1, self.h)

    def get_points(self):
        y = self.generate_y()
        A = self.fit_parameters[0]
        B = self.fit_parameters[1]
        C = self.fit_parameters[2]
        return np.stack((
            A * y ** 2 + B * y + C,
            y
        )).astype(np.int).T

    def evaluate(self):
        y = self.generate_y()
        A = self.fit_parameters[0]
        B = self.fit_parameters[1]
        C = self.fit_parameters[2]
        return A * y ** 2 + B * y + C,

    def curvature_radius(self):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        # Fit new polynomials to x,y in world space
        y = self.get_points()[:, 1] * ym_per_pix
        x = self.get_points()[:, 0] * xm_per_pix
        y_max = 720 * ym_per_pix
        params = np.polyfit(y, x, 2)
        A = params[0]
        B = params[1]

        return int(
            ((1 + (2 * A * y_max + B)**2 )**1.5) /
            np.absolute(2 * A)
        )

    def camera_distance(self):
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        y = self.get_points()
        x = y[np.max(y[:, 1])][0]
        return np.absolute((self.w // 2 - x) * xm_per_pix)