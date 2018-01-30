import numpy as np
import pyqtgraph as pg
from ami.operation.base import Operation, OpConfig


class Projection(Operation):
    def __init__(self, opid, ops):
        super(Projection, self).__init__(opid, ops)
        self.add_input('array')
        self.add_config('axis')

    def run(self):
        self.outputs['proj'] = np.sum(self.array, axis=self.axis)
        return True


class AddArrays(Operation):
    def __init__(self, opid, ops):
        super(AddArrays, self).__init__(opid, ops)
        self.add_input('array1')
        self.add_input('array2')

    def run(self):
        self.outputs['sum'] = self.array1 + self.array2
        return True

class ROI(Operation):
    params = ['shape', 'origin', 'vectors', 'axes']
    def __init__(self, opid, ops):
        super(ROI, self).__init__(opid, ops)
        self.add_input('array')
        self.config.require(*ROI.params)

    @classmethod
    def make_box(cls):
        params = {'outputs' : []}
        for p in ROI.params:
            params[p] = 0
        box = {'config': params, 'opttype': cls.__name__}
        return box

    def run(self):
        self.outputs['roi'] = pg.affineSlice(self.array, self.config['shape'], self.config['origin'], self.config['vectors'], self.config['axes'])
        return True

#class ROIConfig(OpConfig):
#    def __init__(self):
#        super(ROIConfig, self).__init__(['shape','origin'])
