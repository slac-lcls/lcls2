# BlueskyScan.py

from bluesky import RunEngine
from ophyd.status import Status
import sys
import logging
import threading
import zmq
import asyncio
import time
import numpy as np

from psdaq.control.ControlDef import ControlDef

class BlueskyScan:
    def __init__(self, control, *, daqState, args):
        self.control = control
        self.name = 'mydaq'
        self.parent = None
        self.context = zmq.Context()
        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.bind('inproc://bluesky_scan')
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.connect('inproc://bluesky_scan')
        self.comm_thread = threading.Thread(target=self.daq_communicator_thread, args=())
        self.mon_thread = threading.Thread(target=self.daq_monitor_thread, args=(), daemon=True)
        self.ready = threading.Event()
        self.motors = []                # set in configure()
        self.daqState = daqState
        self.daqState_cv = threading.Condition()
        self.stepDone_cv = threading.Condition()
        self.stepDone = 0
        self.comm_thread.start()
        self.mon_thread.start()
        self.verbose = args.v
        self.detname = args.detname
        self.scantype = args.scantype
        self.step_done = threading.Event()
        self.step_value = 1
        self.platform = args.p

        if args.g is None:
            self.groupMask = 1 << self.platform
        else:
            self.groupMask = args.g

        # StepEnd is a cumulative count
        self.readoutCount = args.c
        self.readoutCumulative = 0

    def read(self):
        # stuff we want to give back to user running bluesky
        # e.g. how many events we took.  called when trigger status
        # is done&&successful.
        # typically called by trigger_and_read()
        logging.debug('*** here in read')
        return {}

    def describe(self):
        # stuff we want to give back to user running bluesky
        logging.debug('*** here in describe')
        return {}

    # this thread tells the daq to do a step and waits
    # for the completion, so it can set the bluesky status.
    # it is done as a separate thread so we don't block
    # the bluesky event loop.
    def daq_communicator_thread(self):
        logging.debug('*** daq_communicator_thread')
        while True:
            state = self.pull_socket.recv().decode("utf-8")
            logging.debug('*** received %s' % state)
            if state in ('connected', 'starting'):
                # send 'daqstate(state)' and wait for complete
                # we can block here since we are not in the bluesky
                # event loop
                errMsg = self.control.setState(state)
                if errMsg is not None:
                    logging.error('%s' % errMsg)
                    continue
                with self.daqState_cv:
                    while self.daqState != state:
                        logging.debug('daqState \'%s\', waiting for \'%s\'...' % (self.daqState, state))
                        self.daqState_cv.wait(1.0)
                    logging.debug('daqState \'%s\'' % self.daqState)
                self.ready.set()
            elif state=='running':
                # launch the step with 'daqstate(running)' (with the
                # scan values for the daq to record to xtc2).
                # normally should block on "complete" from the daq here.

                # allow user to override step value
                for motor in self.motors:
                    if motor.name == ControlDef.STEP_VALUE:
                        self.step_value = int(motor.position)
                        logging.debug(f'override: step value = {self.step_value}')
                        break

                my_data = {}

                # record step_value and step_docstring
                my_data.update({'step_value': self.step_value})
                docstring = f'{{"detname": "{self.detname}", "scantype": "{self.scantype}", "step": {self.step_value}}}'
                my_data.update({'step_docstring': docstring})

                # record motor positions
                for motor in self.motors:
                    if motor.name == ControlDef.STEP_VALUE:
                        continue
                    my_data.update({motor.name: motor.position})

                data = {
                  "motors":           my_data,
                  "timestamp":        0,
                  "detname":          self.detname,
                  "dettype":          "scan",
                  "scantype":         self.scantype,
                  "serial_number":    "1234",
                  "alg_name":         "raw",
                  "alg_version":      [1,0,0]
                }

                configureBlock = self.getBlock(transition="Configure", data=data)
                beginStepBlock = self.getBlock(transition="BeginStep", data=data)

                # set DAQ state
                errMsg = self.control.setState('running',
                    {'configure':   {'NamesBlockHex':configureBlock},
                     'beginstep':   {'ShapesDataBlockHex':beginStepBlock},
                     'enable':      {'readout_count':self.readoutCount, 'group_mask':self.groupMask}})
                if errMsg is not None:
                    logging.error('%s' % errMsg)
                    continue

                # wait for running
                with self.daqState_cv:
                    while self.daqState != 'running':
                        logging.debug('daqState \'%s\', waiting for \'running\'...' % self.daqState)
                        self.daqState_cv.wait(1.0)
                    logging.debug('daqState \'%s\'' % self.daqState)

                # wait for step done
                logging.debug('Waiting for step done...')
                self.step_done.wait()
                self.step_done.clear()
                logging.debug(f'step {self.step_value} done.')
                self.step_value += 1

                # tell bluesky step is complete
                # this line is needed in ReadableDevice mode to flag completion
                self.status._finished(success=True)
            elif state=='shutdown':
                break

    def daq_monitor_thread(self):
        logging.debug('*** daq_monitor_thread')
        while True:
            part1, part2, part3, part4, part5, part6, part7, part8 = self.control.monitorStatus()
            if part1 is None:
                break
            elif part1 == 'step':
                self.step_done.set()
                continue
            elif part1 not in ControlDef.transitions:
                continue

            # part1=transition, part2=state, part3=config
            if part1 == 'endrun':
                self.step_value = 1
                logging.debug(f'step value reset to {self.step_value}')

            with self.daqState_cv:
                self.daqState = part2
                self.daqState_cv.notify()

    # for the ReadableDevice style
    # this is a co-routine, so shouldn't block
    def trigger(self):
        # do one step
        self.status = Status()
        # tell the control level to do a step in the scan
        # to-do: pass it the motor positions, ideally both
        # requested/measured positions.
        # maybe don't launch the step directly here
        # with a daqstate command, since it would block
        # the event-loop?
        self.push_socket.send_string('running')     # BeginStep
        self.push_socket.send_string('starting')    # EndStep
        return self.status

    def read_configuration(self):
        # done at the first read after a configure
        return {}

    def describe_configuration(self):
        # the metadata for read_configuration()
        return {}

    # use 'motors' keyword arg to specify a set of motors
    def configure(self, *args, **kwargs):
        logging.debug("*** here in configure")

        if 'motors' in kwargs:
            self.motors = kwargs['motors']
            logging.info('configure: %d motors' % len(self.motors))
        else:
            logging.error('configure: no motors')
        return (self.read_configuration(),self.read_configuration())

    def _set_connected(self):
        self.push_socket.send_string('connected')
        # wait for complete. is this a coroutine, so we shouldn't block?
        self.ready.wait()
        self.ready.clear()

    def stage(self):
        # done once at start of scan
        # put the daq into the right state ('connected')
        logging.debug('*** here in stage')
        self._set_connected()

        return [self]

    def unstage(self):
        # done once at end of scan
        # put the daq into the right state ('connected')
        logging.debug('*** here in unstage')
        self._set_connected()
        
        return [self]

#
# data = {
#   "motors":           {"motor1": 0.0, "step_value": 0.0},
#   "transition":       "Configure",
#   "timestamp":        0,
#   "add_names":        True,
#   "add_shapes_data":  False,
#   "detname":          "scan",
#   "dettype":          "scan",
#   "scantype":         "scan",
#   "serial_number":    "1234",
#   "alg_name":         "raw",
#   "alg_version":      [2,0,0]
# }
#

    def getBlock(self, *, transition, data):
        logging.debug('getBlock: motors=%s' % data["motors"])
        if transition in ControlDef.transitionId.keys():
            data["transitionid"] = ControlDef.transitionId[transition]
        else:
            logging.error(f'invalid transition: {transition}')

        if transition == "Configure":
            data["add_names"] = True
            data["add_shapes_data"] = False
        else:
            data["add_names"] = False
            data["add_shapes_data"] = True

        data["namesid"] = ControlDef.STEPINFO

        return self.control.getBlock(data)
