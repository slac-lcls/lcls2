
"""Simulators of currently non-available LCLS2 data.

   from psana.xtcav.Simulators import SimulatorEBeam, SimulatorGasDetector
   from psana.xtcav.Simulators import SimulatorEventId, SimulatorEnvironment
"""
#----------
import logging
logger = logging.getLogger(__name__)

class Simulator :
    def __init__(self, detname) :
        print('Simulator of %s missing data in LCLS2' % detname)
        self.detname = detname

    def get(self, evt=None) :
        return self

#----------
# for LasingOffReference.py
#----------
import psana.xtcav.Constants as cons

class SimulatorEBeam(Simulator) :
    def __init__(self) :
        Simulator.__init__(self, 'EBeam')


class SimulatorGasDetector(Simulator) :
    def __init__(self) :
        Simulator.__init__(self, 'GasDetector')


class SimulatorEventId(Simulator) :
    def __init__(self) :
        Simulator.__init__(self, 'EventId')


class SimulatorEnvironment(Simulator) :
    def __init__(self) :
        Simulator.__init__(self, 'Environment')

    def calibDir() :
        return './calib'


class SimulatorDetector(Simulator) :
    def __init__(self, name) :
        Simulator.__init__(self, 'Detector')
        self.name = name

    def __call__(self, evt=None, v=1e-100) :
        print('    __call__("%s")' % self.name)
        if   self.name in cons.ROI_SIZE_X_names     : return cons.ROI_SIZE_X
        elif self.name in cons.ROI_SIZE_Y_names     : return cons.ROI_SIZE_Y
        elif self.name in cons.ROI_START_X_names    : return cons.ROI_START_X
        elif self.name in cons.ROI_START_Y_names    : return cons.ROI_START_Y
        elif self.name in cons.UM_PER_PIX_names     : return v
        elif self.name in cons.STR_STRENGTH_names   : return v
        elif self.name in cons.RF_AMP_CALIB_names   : return v
        elif self.name in cons.RF_PHASE_CALIB_names : return v
        elif self.name in cons.DUMP_E_names         : return v
        elif self.name in cons.DUMP_DISP_names      : return v
        elif self.name == cons.ANALYSIS_VERSION     : return None
        else : return None

#----------

if __name__ == "__main__" :
    o0 = Simulator('Superclass for')
    o1 = SimulatorEBeam()
    o2 = SimulatorGasDetector()
    o3 = SimulatorEventId()
    o4 = SimulatorEnvironment()
    o5 = SimulatorDetector(Simulator); print(o5())

#----------
