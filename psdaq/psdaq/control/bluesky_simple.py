# short-term goals:
# - use setstate/monitor stuff
# - get the scan steps into xtc2 (via beginstep phase1 json)
# - make sure psana can read the step info

# questions:
# * how does control.py learn a step is complete?
# * how does scan script learn control.py step is complete?
# - possible ways to complete:
#   - elapsed time
#   - number of events
#   - number of events passing criteria
#   - user setting daqstate in gui?
#   - error
# - how do we count? can't use AMI anymore
#   have Ric return a counter on endstep?
# - maybe timing-system detector handles elapsedTime/events
#   and notifies control? and also records scan info to xtc2?
# - should daq be flyable?  I think answer is no.
#   - if so, is complete called right away and we block until done, or the
#     usual pattern of return a status and mark it done in another thread?
# - who should translate the device-specific scan info (e.g. from
#   motor.read()) into the appropriate dict?
# - should we also store some information about the upcoming scan in beginrun?
#   e.g. scanning vars, numSteps? (although numSteps may not be known)
#   could also extract it quickly from the smd files.  should we store
#   the scan script?

from bluesky import RunEngine
from ophyd.status import Status
import sys
import logging
from psalg.utils.syslog import SysLog
import threading
import zmq
import asyncio
import time
import dgramCreate as dc
import numpy as np

from psdaq.control.control import DaqControl, DaqPVA
import argparse

class MyDAQ:
    def __init__(self, control, motor1, motor2, *, daqState, args):
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
        self.motor1 = motor1
        self.motor2 = motor2
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

        if args.g is None:
            self.groupMask = 1 << args.p
        else:
            self.groupMask = args.g

        # StepEnd is a cumulative count
        self.readoutCount = args.c
        self.readoutCumulative = 0

        # instantiate DaqPVA object
        self.pva = DaqPVA(platform=args.p, xpm_master=args.x, pv_base=args.B)

        # name PVs
        self.pv_xpm_base  = self.pv_base + ':XPM:%d:PART:%d' % (args.x, args.p)
        self.pvStepEnd    = self.pv_xpm_base+':StepEnd'
        self.pvStepDone   = self.pv_xpm_base+':StepDone'

        logging.debug('readoutCount = %s' % self.readoutCount)
        logging.debug('groupMask    = %s' % self.groupMask)
        logging.debug('pvStepEnd    = %s' % self.pvStepEnd)
        logging.debug('pvStepDone   = %s' % self.pvStepDone)

    def read(self):
        # stuff we want to give back to user running bluesky
        # e.g. how many events we took.  called when trigger status
        # is done&&successful.
        # typically called by trigger_and_read()
        logging.debug('*** here in read')
        return {}
        return dict(('channel1',
             {'value': 5, 'timestamp': 1472493713.271991}),
             ('channel2',
             {'value': 16, 'timestamp': 1472493713.539238}))

    def describe(self):
        # stuff we want to give back to user running bluesky
        logging.debug('*** here in describe')
        return {}
        return dict(('channel1',
             {'source': 'XF23-ID:SOME_PV_NAME',
              'dtype': 'number',
              'shape': []}),
            ('channel2',
             {'source': 'XF23-ID:SOME_PV_NAME',
              'dtype': 'number',
              'shape': []}))

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
                self.pva.pv_put(self.pvStepEnd, self.readoutCumulative)
                self.pva.step_groups(mask=self.groupMask)
                self.pva.pv_put(self.pvStepDone, 0)
                with self.stepDone_cv:
                    self.stepDone = 0
                    self.stepDone_cv.notify()

                # set DAQ state
                errMsg = self.control.setState('running',
                    {'configure':{'NamesBlockHex':self.getBlock(transitionid=DaqControl.transitionId['Configure'],
                                                                add_names=True,
                                                                add_shapes_data=False).hex()},
                     'beginstep':{'ShapesDataBlockHex':self.getBlock(transitionid=DaqControl.transitionId['BeginStep'],
                                                                add_names=False,
                                                                add_shapes_data=True).hex()}})
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
            part1, part2, part3, part4 = self.control.monitorStatus()
            if part1 is None:
                break
            elif part1 not in DaqControl.transitions:
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

    def configure(self, *args, **kwargs):
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

    def getBlock(self, *, transitionid, add_names, add_shapes_data):
        my_data = {
            self.motor1.name: self.motor1.position,
            self.motor2.name: self.motor2.position
        }

        detname       = 'scan'
        dettype       = 'scan'
        serial_number = '1234'
        namesid       = 253     # STEPINFO = 253 (psdaq/drp/drp.hh)
        nameinfo      = dc.nameinfo(detname,dettype,serial_number,namesid)

        alg           = dc.alg('raw',[2,0,0])

        self.cydgram.addDet(nameinfo, alg, my_data)

        # create dgram
        timestamp    = 0
        xtc_bytes    = self.cydgram.getSelect(timestamp, transitionid, add_names=add_names, add_shapes_data=add_shapes_data)
        logging.debug('transitionid %d dgram is %d bytes (with header)' % (transitionid, len(xtc_bytes)))

        # remove first 12 bytes (dgram header), and keep next 12 bytes (xtc header)
        return xtc_bytes[12:]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-B', metavar='PVBASE', required=True, help='PV base')
    parser.add_argument('-p', type=int, choices=range(0, 8), default=0,
                        help='platform (default 0)')
    parser.add_argument('-x', metavar='XPM', type=int, required=True, help='master XPM')
    parser.add_argument('-C', metavar='COLLECT_HOST', default='localhost',
                        help='collection host (default localhost)')
    parser.add_argument('-t', type=int, metavar='TIMEOUT', default=10000,
                        help='timeout msec (default 10000)')
    parser.add_argument('-c', type=int, metavar='READOUT_COUNT', default=1, help='# of events to aquire at each step (default 1)')
    parser.add_argument('-g', type=int, metavar='GROUP_MASK', help='bit mask of readout groups (default 1<<plaform)')
    parser.add_argument('--config', metavar='ALIAS', help='configuration alias (e.g. BEAM)')
    parser.add_argument('-v', action='store_true', help='be verbose')
    args = parser.parse_args()

    if args.g is not None:
        if args.g < 1 or args.g > 255:
            parser.error('readout group mask (-g) must be 1-255')

    if args.c < 1:
        parser.error('readout count (-c) must be >= 1')

    # instantiate DaqControl object
    control = DaqControl(host=args.C, platform=args.p, timeout=args.t)

    # configure logging handlers
    instrument = control.getInstrument()
    if args.v:
        level=logging.DEBUG
    else:
        level=logging.WARNING
    logger = SysLog(instrument=instrument, level=level)
    logging.info('logging initialized')

    # get initial DAQ state
    daqState = control.getState()
    logging.info('initial state: %s' % daqState)
    if daqState == 'error':
        sys.exit(1)

    # optionally set BEAM or NOBEAM
    if args.config is not None:
        # config alias request
        rv = control.setConfig(args.config)
        if rv is not None:
            logging.error('%s' % rv)

    RE = RunEngine({})

    # cpo thinks this is more for printout of each step
    from bluesky.callbacks.best_effort import BestEffortCallback
    bec = BestEffortCallback()

    # Send all metadata/data captured to the BestEffortCallback.
    RE.subscribe(bec)

    from ophyd.sim import motor1, motor2
    from bluesky.plans import scan

    # instantiate MyDAQ object
    mydaq = MyDAQ(control, motor1, motor2, daqState=daqState, args=args)
    dets = [mydaq]   # just one in this case, but it could be more than one

    # Scan motor1 from -10 to 10 and motor2 from -0.1 to 0.1, stopping
    # at 15 equally-spaced points along the way and reading dets.
    RE(scan(dets, motor1, -10, 10, motor2, -0.1, 0.1, 15))

    mydaq.push_socket.send_string('shutdown') #shutdown the daq thread
    mydaq.comm_thread.join()

if __name__ == '__main__':
    main()
