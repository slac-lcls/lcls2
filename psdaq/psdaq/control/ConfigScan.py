import logging
import zmq
from threading import Thread, Event, Condition
import json as oldjson
import dgramCreate as dc
from psdaq.control.ControlDef import ControlDef, front_pub_port, front_rep_port, create_msg

class ConfigScan:
    def __init__(self, control, *, daqState, args):
        self.zmq_port = 5550+args.p     # one port per platform
        self.control = control
        self.name = 'mydaq'
        self.parent = None
        self.context = zmq.Context()
        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.bind('tcp://*:%s' % self.zmq_port)
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.connect('tcp://localhost:%s' % self.zmq_port)
        self.comm_thread = Thread(target=self.daq_communicator_thread, args=())
        self.mon_thread = Thread(target=self.daq_monitor_thread, args=(), daemon=True)
        self.ready = Event()
        self.step_done = Event()
        self.daqState = daqState
        self.args = args
        self.daqState_cv = Condition()
        self.stepDone_cv = Condition()
        self.stepDone = 0
        self.comm_thread.start()
        self.mon_thread.start()
        self.verbose = args.v
        self.motors = []                # set in configure()
        self._step_count = 0
        self.cydgram = dc.CyDgram()

        if args.g is None:
            self.groupMask = 1 << args.p
        else:
            self.groupMask = args.g

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
                    errMsg = self.control.setState(state, oldjson.loads(phase1))

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

                # set DAQ state
                if phase1 is None:
                    errMsg = self.control.setState(state)
                else:
                    errMsg = self.control.setState(state, oldjson.loads(phase1))
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
                logging.debug('step done.')

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
        self._set_connected()
        self._step_count = 0

    def unstage(self):
        # done once at end of scan
        # put the daq into the right state ('connected')
        logging.debug('*** unstage: step count = %d' % self._step_count)
        self._set_connected()

    def getBlock(self, *, transitionid, add_names, add_shapes_data):
        my_data = {}
        for motor in self.motors:
            my_data.update({motor.name: motor.position})

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


    # use 'motors' keyword arg to specify a set of motors
    def configure(self, *args, **kwargs):
        logging.debug("*** here in configure")

        if 'motors' in kwargs:
            self.motors = kwargs['motors']
            logging.info('configure: %d motors' % len(self.motors))
        else:
            logging.error('configure: no motors')

    def step_count(self):
        return self._step_count

    def update(self, *, value):
        # update 'motors'
        for motor in self.motors:
            motor.update(value)

    def trigger(self, *, phase1Info=None):
        # do one step
        logging.debug('*** trigger: step count = %d' % self._step_count)
        if phase1Info is None:
            phase1Info = {}
        if "beginstep" not in phase1Info:
            phase1Info.update({"beginstep": {}})
        if "configure" not in phase1Info:
            phase1Info.update({"configure": {}})
        if "step_keys" not in phase1Info["configure"]:
            phase1Info["configure"].update({"step_keys": []})
        if "step_values" not in phase1Info["beginstep"]:
            phase1Info["beginstep"].update({"step_values": {}})

        logging.debug('*** phase1Info = %s' % oldjson.dumps(phase1Info))
        # BeginStep
        self.push_socket.send_string('running,%s' % oldjson.dumps(phase1Info))
        # EndStep
        self.push_socket.send_string('starting')
        self._step_count += 1

