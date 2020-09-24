
"""
Module :py:class:`CGDaqControl` contains proxy and singleton for psdaq/control/control.py
=============================================================================================

Usage ::

    from psdaq.control_gui.CGDaqControl import daq_control, DaqControl #, DaqControlEmulator

    print('DaqControl.transitions:', DaqControl.transitions)
    print('DaqControl.states  :', DaqControl.states)

    daq_control.set_daq_control(DaqControl(host='localhost', platform=2, timeout=10000))
    #daq_control.set_daq_control(DaqControlEmulator())

    o = get_daq_control() # safe for daq_control()

    daq_control().setstate('running') # DaqControl.states[5]
    # or    
    daq_control_set_state('configured')

    state = daq_control().getState()
    # or
    state = daq_control_get_state()

    transition, state, cfgtype, recording, platform, bypass_activedet, experiment_name, run_number, last_run_number = daq_control_get_status()
    instr = daq_control_get_instrument()

    daq_control_set_record(do_record=True)

See:
    - :class:`CGDaqControl`
    - `graphqt documentation <https://lcls-psana.github.io/graphqt/py-modindex.html>`_.

Created on 2019-02-01 by Mikhail Dubrovin
"""

#----------

import logging
logger = logging.getLogger(__name__)
from time import time
from psdaq.control.control import DaqControl

#----------

class Emulator :
    def __init__(self) :
        #from psdaq.control_gui.CGConfigParameters import cp
        #self.wpart = self # cp.cgwmainpartition
        pass

    def set_buts_enable(self, s) :
        pass

#----------

class DaqControlEmulator:
    """Emulates interaction with DaqControl, DO NOT DO ANYTHING, prints warning messages.
    """
    def __init__(self) :
        self._name = 'DaqControlEmulator'
    def msg(self, s) : logger.warning('TEST PURPOSE ONLY DaqControlEmulator.%s' % s) 
    def getInstrument(self) :     self.msg('getInstrument');  return 'EMU'
    def setConfig(self, c) :      self.msg('setConfig %s'% str(c)); return None
    def setState(self, s) :       self.msg('setState %s'%s);
    def getState(self) :          self.msg('getState');       return 'emulator'
    def getStatus(self) :         self.msg('getStatus');      return 'running', 'running', 'BEAM', False, {}, False, 'exp123456', 2, 1
    def setTransition(self, s) :  self.msg('setTransition');  return 'emulator' 
    def selectPlatform(self, s) : self.msg('selectPlatform'); return
    def getPlatform(self) :       self.msg('getPlatform');    return 'emulator'
    def setRecord(self, v) :      self.msg('setRecord %s'%v); return

#----------

class DaqControlProxy:
  def __init__(self, o=None) :
      """Creates proxy object, e.g. for singleton.
      """
      self.o = None
      self.set_daq_control(o)

  def set_daq_control(self, o=None) :
      """Sets object whenever it is available, Nobe by default.
      """
      if self.o is not None : del self.o
      self.o = o

  def __call__(self) :
      """Access object, e.g. using singleton:
         state = daq_control().getstate()
      """
      return self.o

#---------- SINGLETON

daq_control = DaqControlProxy()

def worker_set_state(dicio):
    state = dicio.get('state_in','N/A')
    logger.debug('worker_set_state %s' % state)
    daq_control().setState(state)
       
def worker_get_state(dicio):
    dicio['state_out'] = daq_control().getState()

#----------

def get_daq_control(cmt='') :
    daq_ctrl = daq_control()
    if daq_ctrl is None :
        logger.warning('%sdaq_control() is None' % cmt)
    return daq_ctrl

#----------

def daq_control_set_state(s='configured') :
    daq_ctrl = get_daq_control('in daq_control_set_state ')
    if daq_ctrl is None : return False

    daq_ctrl.setState(s)
    logger.debug('daq_control_set_state("%s")' % s)
    return True

#----------

def daq_control_get_state() :
    daq_ctrl = get_daq_control('in daq_control_get_state ')
    if daq_ctrl is None : return None
    s = daq_ctrl.getState()
    logger.debug('daq_control_get_state(): %s' % s)
    return s

#----------

def daq_control_get_instrument() :
    daq_ctrl = get_daq_control('in daq_control_get_instrument ')
    if daq_ctrl is None : return None
    s = daq_ctrl.getInstrument()
    logger.debug('daq_control_get_instrument(): %s' % s)
    return s

#----------

def daq_control_get_status() :

    #t0_sec = time()
    daq_ctrl = get_daq_control('in daq_control_get_status ')
    if daq_ctrl is None : return None

    transition, state, cfgtype, recording, platform, bypass_activedet, \
        experiment_name, run_number, last_run_number = daq_ctrl.getStatus()
    logger.debug('daq_control_get_status transition:%s state:%s cfgtype:%s recording:%s bypass_activedet:%s'%\
                 (str(transition), str(state), str(cfgtype), str(recording), str(bypass_activedet)))
    logger.debug('daq_control_get_status experiment_name:%s run_number:%s last_run_number:%s'%\
                 (str(experiment_name), str(run_number), str(last_run_number)))
    logger.debug('daq_control_get_status platform:%s' % str(platform))

    #logger.debug('daq_control_get_status() time = %.6f sec' % (time()-t0_sec))

    return transition.lower(), state.lower(), cfgtype, recording, platform, \
        bypass_activedet, experiment_name, run_number, last_run_number

#----------

def daq_control_set_record(do_record=True) :
    daq_ctrl = get_daq_control('in daq_control_set_record ')
    if daq_ctrl is None : return False

    daq_ctrl.setRecord(do_record) # sets flag for recording
    logger.debug('daq_control_set_record("%s")' % (do_record))
    return True

#----------
#----------
#----------
#----------
#----------
#----------

if __name__ == "__main__" :
  def proc() :
    print('DaqControl.transitions:', DaqControl.transitions)
    print('DaqControl.states  :', DaqControl.states)

    o = DaqControl(host='localhost', platform=6, timeout=5000) # msec
    daq_control.set_daq_control(o)

    state = daq_control().getState()
    print('DaqControl.states  :', state)

#----------

if __name__ == "__main__" :
    proc()

#----------
