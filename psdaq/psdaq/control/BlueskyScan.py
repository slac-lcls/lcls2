# BlueskyScan.py

from bluesky import RunEngine
from ophyd.status import Status
import sys
import logging
import threading
import zmq
import asyncio
import time
import dgramCreate as dc
import numpy as np

from psdaq.control.ControlDef import ControlDef
from psdaq.control.control import DaqPVA

class BlueskyScan:
    def __init__(self, control, *, daqState, args):
        self.control = control
        self.name = 'mydaq'
        self.parent = None
        self.context = zmq.Context()
        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.bind('tcp://*:5555')
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.connect('tcp://localhost:5555')
        self.comm_thread = threading.Thread(target=self.daq_communicator_thread, args=())
        self.mon_thread = threading.Thread(target=self.daq_monitor_thread, args=(), daemon=True)
        self.ready = threading.Event()
        self.motors = []                # set in configure()
        self.cydgram = dc.CyDgram()
        self.daqState = daqState
        self.args = args
        self.daqState_cv = threading.Condition()
        self.stepDone_cv = threading.Condition()
        self.stepDone = 0
        self.comm_thread.start()
        self.mon_thread.start()
        self.verbose = args.v
        self.pv_base = args.B
        self.detname = args.detname
        self.scantype = args.scantype

        if args.g is None:
            self.groupMask = 1 << args.p
        else:
            self.groupMask = args.g

        # StepEnd is a cumulative count
        self.readoutCount = args.c
        self.readoutCumulative = 0

        # instantiate DaqPVA object
        self.pva = DaqPVA(platform=args.p, xpm_master=args.x, pv_base=args.B)

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

                # set EPICS PVs.
                # StepEnd is a cumulative count.
                self.readoutCumulative += self.readoutCount
                self.pva.pv_put(self.pva.pvStepEnd, self.readoutCumulative)
                self.pva.step_groups(mask=self.groupMask)
                self.pva.pv_put(self.pva.pvStepDone, 0)
                with self.stepDone_cv:
                    self.stepDone = 0
                    self.stepDone_cv.notify()

                my_data = {}
                for motor in self.motors:
                    my_data.update({motor.name: motor.position})
                    # derive step_docstring from step_value
                    if motor.name == 'step_value':
                        docstring = f'{{"detname": "{self.detname}", "scantype": "{self.scantype}", "step": {motor.position}}}'
                        my_data.update({'step_docstring': docstring})

                data = {
                  "motors":           my_data,
                  "timestamp":        0,
                  "detname":          "scan",
                  "dettype":          "scan",
                  "scantype":         "scan",
                  "serial_number":    "1234",
                  "alg_name":         "raw",
                  "alg_version":      [2,0,0]
                }

                configureBlock = self.getBlock(transition="Configure", data=data)
                beginStepBlock = self.getBlock(transition="BeginStep", data=data)

                # set DAQ state
                errMsg = self.control.setState('running',
                    {'configure':{'NamesBlockHex':configureBlock.hex()},
                     'beginstep':{'ShapesDataBlockHex':beginStepBlock.hex()}})
                if errMsg is not None:
                    logging.error('%s' % errMsg)
                    continue

                with self.daqState_cv:
                    while self.daqState != 'running':
                        logging.debug('daqState \'%s\', waiting for \'running\'...' % self.daqState)
                        self.daqState_cv.wait(1.0)
                    logging.debug('daqState \'%s\'' % self.daqState)

                # define nested function for monitoring the StepDone PV
                def callback(stepDone):
                    with self.stepDone_cv:
                        self.stepDone = int(stepDone)
                        self.stepDone_cv.notify()

                # start monitoring the StepDone PV
                sub = self.pva.monitor_StepDone(callback=callback)

                with self.stepDone_cv:
                    while self.stepDone != 1:
                        logging.debug('PV \'StepDone\' is %d, waiting for 1...' % self.stepDone)
                        self.stepDone_cv.wait(1.0)
                logging.debug('PV \'StepDone\' is %d' % self.stepDone)

                # stop monitoring the StepDone PV
                sub.close()

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
            elif part1 not in ControlDef.transitions:
                continue

            # part1=transition, part2=state, part3=config
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
        if transition == "Configure":
            data["add_names"] = True
            data["add_shapes_data"] = False
        else:
            data["add_names"] = False
            data["add_shapes_data"] = True
        detname       = data["detname"]
        dettype       = data["dettype"]
        serial_number = data["serial_number"]
        namesid       = ControlDef.STEPINFO
        nameinfo      = dc.nameinfo(detname,dettype,serial_number,namesid)
        alg_name      = data["alg_name"]
        alg_version   = data["alg_version"]
        alg           = dc.alg(data["alg_name"], data["alg_version"])
        self.cydgram.addDet(nameinfo, alg, data["motors"])

        # create dgram
        add_names       = data["add_names"]
        add_shapes_data = data["add_shapes_data"]
        timestamp       = data["timestamp"]
        transitionid    = ControlDef.transitionId[transition]

        xtc_bytes    = self.cydgram.getSelect(timestamp, transitionid, add_names=add_names, add_shapes_data=add_shapes_data)
        logging.debug('getBlock: transitionid %d dgram is %d bytes (with header)' % (transitionid, len(xtc_bytes)))

        # remove first 12 bytes (dgram header), and keep next 12 bytes (xtc header)
        return xtc_bytes[12:]
