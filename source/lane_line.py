import numpy as np

class LaneLine(object):

    def __init__(self, image, window_points):
        self.points = np.zeros_like(image)
        self.points[(window_points == 255) & (image == 1)] = 1
        self.x = np.where(self.points == 1)[1]
        self.y = np.where(self.points == 1)[0]
        self.fit_parameters = None

    def fit(self):
        self.fit_parameters = np.polyfit(self.y, self.x, deg=2)
        return self.fit_parameters

    def evaluate(self, y):
        return self.fit_parameters[0] * y**2 + \
               self.fit_parameters[1] * y**1 + \
               self.fit_parameters[2] * y**0
