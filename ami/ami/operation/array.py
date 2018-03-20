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


class Sum(OpConfig):
    def operate(self, a, b):
        return a + b


class Scale(OpConfig):
    # maybe should just be "Multiply"?
    def operate(self, inpt, scale_factor):
        return inpt * scale_factor


class ROI(OpConfig):
    def __init__(self, shape, vector, origin, axes):
        super(__class__, self).__init__("shape", "vector", "origin", "axes")
        self.shape = shape
        self.vector = vector
        self.origin = origin
        self.axes = axes

    def operate(self, image):
       return pg.affineSlice(image, self.shape, self.origin, self.vector, self.axes)
