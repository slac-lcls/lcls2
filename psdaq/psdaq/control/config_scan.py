import sys
import logging
import dgramCreate as dc
import threading
import zmq
import json

from psdaq.control.control import ControlDef, DaqPVA, DaqXPM
from psdaq.control.DaqControl import DaqControl
import argparse

class MyDAQ:
    def __init__(self, control, *, daqState, args):
        self.zmq_port = 5550+args.p
        self.control = control
        self.name = 'mydaq'
        self.parent = None
        self.context = zmq.Context()
        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.bind('tcp://*:%s' % self.zmq_port)
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.connect('tcp://localhost:%s' % self.zmq_port)
        self.comm_thread = threading.Thread(target=self.daq_communicator_thread, args=())
        self.mon_thread = threading.Thread(target=self.daq_monitor_thread, args=(), daemon=True)
        self.ready = threading.Event()
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
        self.pva = DaqPVA(report_error=self.report_error)
        self.xpm = DaqXPM(platform=args.p, xpm_master=args.x, pv_base=args.B, pva=self.pva, zctxt=self.context, xpm_host=args.X, report_error=self.report_error)

    # this thread tells the daq to do a step and waits for the completion
    def daq_communicator_thread(self):
        logging.debug('*** daq_communicator_thread')
        while True:
            sss = self.pull_socket.recv().decode("utf-8")
            if ',' in sss:
                state, phase1 = sss.split(',', maxsplit=1)
            else:
                state, phase1 = sss, None

            logging.debug('*** received %s' % state)
            if state in ('connected', 'starting'):
                # send 'daqstate(state)' and wait for complete
                if phase1 is None:
                    errMsg = self.control.setState(state)
                else:
                    errMsg = self.control.setState(state, json.loads(phase1))

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

                # set EPICS PVs.
                # StepEnd is a cumulative count.
                self.readoutCumulative += self.readoutCount
                #self.pva.pv_put(self.xpm.pvStepEnd, self.readoutCumulative)
                #self.xpm.step_groups(mask=self.groupMask)
                #self.pva.pv_put(self.xpm.pvStepDone, 0)
                self.xpm.setup_step(self.xpm.platform,self.groupMask,self.readoutCumulative)
                with self.stepDone_cv:
                    self.stepDone = 0
                    self.stepDone_cv.notify()

                # set DAQ state
                if phase1 is None:
                    errMsg = self.control.setState(state)
                else:
                    errMsg = self.control.setState(state, json.loads(phase1))
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
                sub = self.xpm.monitor_StepDone(callback=callback)

                with self.stepDone_cv:
                    while self.stepDone != 1:
                        logging.debug('PV \'StepDone\' is %d, waiting for 1...' % self.stepDone)
                        self.stepDone_cv.wait(1.0)
                logging.debug('PV \'StepDone\' is %d' % self.stepDone)

                # stop monitoring the StepDone PV
                sub.close()

                self.ready.set()

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

    def _set_connected(self):
        self.push_socket.send_string('connected')
        # wait for complete
        self.ready.wait()
        self.ready.clear()

    def stage(self):
        # done once at start of scan
        # put the daq into the right state ('connected')
        logging.debug('*** here in stage')
        self._set_connected()

    def unstage(self):
        # done once at end of scan
        # put the daq into the right state ('connected')
        logging.debug('*** here in unstage')
        self._set_connected()

    def trigger(self, *, phase1Info=None):
        # do one step
        logging.debug('*** here in trigger')
        if phase1Info is None:
            # BeginStep
            self.push_socket.send_string('running')
        else:
            logging.debug('*** phase1Info = %s' % json.dumps(phase1Info))
            # BeginStep
            self.push_socket.send_string('running,%s' % json.dumps(phase1Info))

        # EndStep
        self.ready.wait()
        self.ready.clear()
        self.push_socket.send_string('starting')
        self.ready.wait()
        self.ready.clear()

    def report_error(self, msg):
        logging.error(msg)
        return

def scan( keys, steps, defargs={}, setupStep=None ):
    parser = argparse.ArgumentParser()
    def add_argument(arg,metavar='',default=None,required=False,help='',**kwargs):
        if arg in defargs:
            default = defargs[arg]
            required = False
        parser.add_argument(arg,default=default,metavar=metavar,required=required,help=help,**kwargs)

    add_argument('-B', metavar='PVBASE', required=True, help='PV base')
    add_argument('-p', type=int, choices=range(0, 8), default=0, help='platform (default 0)')
    add_argument('-x', metavar='XPM', type=int, required=True, help='master XPM')
    add_argument('-X', metavar='XPMHOST', type=str, default=None)
    add_argument('-C', metavar='COLLECT_HOST', default='localhost',
                        help='collection host (default localhost)')
    add_argument('-t', type=int, metavar='TIMEOUT', default=10000,
                        help='timeout msec (default 10000)')
    add_argument('-c', type=int, metavar='READOUT_COUNT', default=1, help='# of events to aquire at each step (default 1)')
    add_argument('-g', type=int, metavar='GROUP_MASK', help='bit mask of readout groups (default 1<<plaform)')
    add_argument('--config', metavar='ALIAS', help='configuration alias (e.g. BEAM)')
    add_argument('--record', type=int, choices=range(0, 2), help='recording flag')
    parser.add_argument('-v', action='store_true', help='be verbose')

    args = parser.parse_args()

    if args.g is not None:
        if args.g < 1 or args.g > 255:
            parser.error('readout group mask (-g) must be 1-255')

    if args.c < 1:
        parser.error('readout count (-c) must be >= 1')

    # instantiate DaqControl object
    control = DaqControl(host=args.C, platform=args.p, timeout=args.t)

    try:
        instrument = control.getInstrument()
    except KeyboardInterrupt:
        instrument = None

    if instrument is None:
        sys.exit('Error: failed to read instrument name (check -C <COLLECT_HOST>)')

    # configure logging handlers
    if args.v:
        level=logging.DEBUG
    else:
        level=logging.WARNING
    logging.basicConfig(level=level)
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

    if args.record is not None:
        # recording flag request
        if args.record == 0:
            rv = control.setRecord(False)
        else:
            rv = control.setRecord(True)
        if rv is not None:
            print('Error: %s' % rv)

    # instantiate MyDAQ
    mydaq = MyDAQ(control, daqState=daqState, args=args)

    mydaq.stage()

    # -- begin script --------------------------------------------------------

    configure_dict = {"configure": {"step_keys": keys,
                                    "NamesBlockHex":scanNamesBlock()}}

    for step in steps():
        if setupStep is not None:
            setupStep(step)

        beginstep_dict = {"beginstep": {"step_values": step[0],
                                        "ShapesDataBlockHex":shapesDataBlock(step)}}
        # trigger
        mydaq.trigger(phase1Info = dict(configure_dict, **beginstep_dict))

    # -- end script ----------------------------------------------------------

    mydaq.unstage()

    mydaq.push_socket.send_string('shutdown') #shutdown the daq communicator thread
    mydaq.comm_thread.join()


def setupXtc(step=None):
    d = {'step_value'    :0.,
         'step_docstring':''}
    if step and len(step)==3:
        d['step_value'    ]=step[1]
        d['step_docstring']=step[2]
    print(f'setupXtc {d}')

    nameinfo = dc.nameinfo('scan','scan','1234',253)
    alg      = dc.alg('raw',[2,0,0])
    cydgram  = dc.CyDgram()
    cydgram.addDet(nameinfo, alg, d)
    return cydgram

def scanNamesBlock():
    cydgram = setupXtc()
    return cydgram.getSelect(0, ControlDef.transitionId['Configure'],
                             add_names=True, add_shapes_data=False)[12:].hex()

def shapesDataBlock(step):
    cydgram = setupXtc(step)
    return cydgram.getSelect(0, ControlDef.transitionId['BeginStep'],
                             add_names=False, add_shapes_data=True)[12:].hex()


