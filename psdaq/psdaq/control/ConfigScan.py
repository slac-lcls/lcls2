import logging
import zmq
from threading import Thread, Event, Condition
import json as oldjson
from psdaq.control.ControlDef import ControlDef, front_pub_port, front_rep_port, create_msg

class ConfigScan:
    def __init__(self, control, *, daqState, args):
        self.control = control
        self.name = 'mydaq'
        self.parent = None
        self.context = zmq.Context()
        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.bind('inproc://config_scan')
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.connect('inproc://config_scan')
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
        self.detname = args.detname
        self.scantype = args.scantype

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

    # use 'motors' keyword arg to specify a set of motors
    def configure(self, *args, **kwargs):
        logging.debug("*** here in configure")

        if 'motors' in kwargs:
            self.motors = kwargs['motors']
            logging.info('configure: %d motors' % len(self.motors))
        else:
            logging.error('configure: no motors')

    def getMotors(self):
        return self.motors

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
