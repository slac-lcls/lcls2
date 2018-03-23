import numpy as np
import pyqtgraph as pg
from ami.data import DataTypes
from ami.operation.base import OpConfig, OperationError

class Projection(OpConfig):
    def __init__(self, axis):
        super(__class__, self).__init__("axis")
        self.axis = axis

    def operate(self, image):
        return np.sum(image, axis=self.axis)


class SumScalar(OpConfig):
    def __init__(self):
        super(__class__, self).__init__()

    def operate(self, value1, value2):
        return value1 + value2


class SumImage(OpConfig):
    def __init__(self):
        super(__class__, self).__init__()

    def operate(self, image1, image2):
        return image1 + image2


class SumWaveform(OpConfig):
    def __init__(self):
        super(__class__, self).__init__()

    def operate(self, wave1, wave2):
        return wave1 + wave2


class ROI(OpConfig):
    def __init__(self, shape, vector, origin, axes):
        super(__class__, self).__init__("shape", "vector", "origin", "axes")
        self.shape = shape
        self.vector = vector
        self.origin = origin
        self.axes = axes

    def operate(self, image):
       return pg.affineSlice(image, self.shape, self.origin, self.vector, self.axes)
