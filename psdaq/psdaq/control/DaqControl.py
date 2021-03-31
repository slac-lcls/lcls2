import logging
import zmq
from psdaq.control.DaqDefs import DaqDefs, front_pub_port, front_rep_port, create_msg

class DaqControl:
    'Base class for controlling data acquisition'

    # default readout group is self.platform

    def __init__(self, *, host, platform, timeout):
        self.host = host
        self.platform = platform
        self.timeout = timeout

        # initialize zmq socket
        self.context = zmq.Context(1)
        self.front_sub = self.context.socket(zmq.SUB)
        self.front_sub.connect('tcp://%s:%d' % (host, front_pub_port(platform)))
        self.front_sub.setsockopt(zmq.SUBSCRIBE, b'')
        self.front_req = None
        self.front_req_endpoint = 'tcp://%s:%d' % (host, front_rep_port(platform))
        self.front_req_init()

    #
    # DaqControl.getState - get current state
    #
    def getState(self):
        retval = 'error'
        try:
            msg = create_msg('getstate')
            self.front_req.send_json(msg)
            reply = self.front_req.recv_json()
        except zmq.Again:
            logging.error('getState() timeout (%.1f sec)' % (self.timeout / 1000.))
            logging.info('getState() reinitializing zmq socket')
            self.front_req_init()
        except Exception as ex:
            logging.error('getState() Exception: %s' % ex)
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        else:
            try:
                retval = reply['header']['key']
            except KeyError:
                pass

        return retval

    #
    # DaqControl.getPlatform - get platform
    #
    def getPlatform(self):
        retval = {}
        try:
            msg = create_msg('getstate')
            self.front_req.send_json(msg)
            reply = self.front_req.recv_json()
        except zmq.Again:
            logging.error('getPlatform() timeout (%.1f sec)' % (self.timeout / 1000.))
            logging.info('getPlatform() reinitializing zmq socket')
            self.front_req_init()
        except Exception as ex:
            logging.error('getPlatform() Exception: %s' % ex)
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        else:
            try:
                retval = reply['body']
            except KeyError:
                pass

        return retval

    #
    # DaqControl.getJsonConfig - get json configuration
    #
    def getJsonConfig(self):
        src = self.getPlatform()
        dst = levels_to_activedet(src)
        retval =  oldjson.dumps(dst, sort_keys=True, indent=4)
        logging.debug('getJsonConfig() return value: %s' % retval)
        return retval

    #
    # DaqControl.storeJsonConfig - store json configuration
    #
    def storeJsonConfig(self, json_data):
        retval = {}
        body = {"json_data": json_data}
        try:
            msg = create_msg('storejsonconfig', body=body)
            self.front_req.send_json(msg)
            reply = self.front_req.recv_json()
        except zmq.Again:
            logging.error('storeJsonConfig() timeout (%.1f sec)' % (self.timeout / 1000.))
            logging.info('storeJsonConfig() reinitializing zmq socket')
            self.front_req_init()
        except Exception as ex:
            logging.error('storeJsonConfig() Exception: %s' % ex)
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        else:
            try:
                retval = reply['body']
            except KeyError:
                pass

        return retval

    #
    # DaqControl.selectPlatform - select platform
    #
    def selectPlatform(self, body):
        retval = {}
        try:
            msg = create_msg('selectplatform', body=body)
            self.front_req.send_json(msg)
            reply = self.front_req.recv_json()
        except zmq.Again:
            logging.error('selectPlatform() timeout (%.1f sec)' % (self.timeout / 1000.))
            logging.info('selectPlatform() reinitializing zmq socket')
            self.front_req_init()
        except Exception as ex:
            logging.error('selectPlatform() Exception: %s' % ex)
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        else:
            try:
                retval = reply['body']
            except KeyError:
                pass

        return retval

    #
    # DaqControl.getInstrument - get instrument name
    #
    def getInstrument(self):
        r1 = None
        try:
            msg = create_msg('getinstrument')
            self.front_req.send_json(msg)
            reply = self.front_req.recv_json()
        except Exception as ex:
            print('getInstrument() Exception: %s' % ex)
        else:
            try:
                r1 = reply['body']['instrument']
            except Exception as ex:
                print('getInstrument() Exception: %s' % ex)

        return r1

    #
    # DaqControl.getStatus - get status
    #
    def getStatus(self):
        r1 = r2 = r3 = r4 = r6 = 'error'
        r5 = {}
        r7 = r8 = r9 = 'error'
        try:
            msg = create_msg('getstatus')
            self.front_req.send_json(msg)
            reply = self.front_req.recv_json()
        except Exception as ex:
            print('getStatus() Exception: %s' % ex)
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        else:
            try:
                r1 = reply['body']['transition']
                r2 = reply['body']['state']
                r3 = reply['body']['config_alias']
                r4 = reply['body']['recording']
                r5 = reply['body']['platform']
                r6 = reply['body']['bypass_activedet']
                r7 = reply['body']['experiment_name']
                r8 = reply['body']['run_number']
                r9 = reply['body']['last_run_number']
            except KeyError:
                pass

        return (r1, r2, r3, r4, r5, r6, r7, r8, r9)

    #
    # DaqControl.monitorStatus - monitor the status
    #
    def monitorStatus(self):

        # process messages
        while True:
            try:
                msg = self.front_sub.recv_json()

                if msg['header']['key'] == 'status':
                    # return transition, state, config_alias, recording, bypass_activedet, experiment_name, run_number, last_run_number
                    return msg['body']['transition'], msg['body']['state'], msg['body']['config_alias'], msg['body']['recording'], msg['body']['bypass_activedet'],\
                           msg['body']['experiment_name'], msg['body']['run_number'], msg['body']['last_run_number']

                elif msg['header']['key'] == 'error':
                    # return 'error', error message, 'error', 'error', 'error', 'error', 'error', 'error'
                    return 'error', msg['body']['err_info'], 'error', 'error', 'error', 'error', 'error', 'error'

                elif msg['header']['key'] == 'warning':
                    # return 'error', error message, 'error', 'error', 'error', 'error', 'error', 'error'
                    return 'warning', msg['body']['err_info'], 'error', 'error', 'error', 'error', 'error', 'error'

                elif msg['header']['key'] == 'fileReport':
                    # return 'fileReport', path, 'error', 'error', 'error', 'error', 'error', 'error'
                    return 'fileReport', msg['body']['path'], 'error', 'error', 'error', 'error', 'error', 'error'

                elif msg['header']['key'] == 'progress':
                    # return 'progress', transition, elapsed, total, 'error', 'error', 'error', 'error'
                    return 'progress', msg['body']['transition'], msg['body']['elapsed'], msg['body']['total'], 'error', 'error', 'error', 'error'

            except KeyboardInterrupt:
                break

            except KeyError as ex:
                logging.error('KeyError: %s' % ex)
                break

        return None, None, None, None, None, None, None, None

    #
    # DaqControl.setState - change the state
    # The optional second argument is a dictionary containing
    # one entry per transition that contains information that
    # will be put into the phase1-json of the transition. An example:
    # {'beginstep': {'myvalue1':3 , 'myvalue2': {'myvalue3':72}},
    #  'enable':    {'myvalue5':37, 'myvalue6': 'hello'}}
    #
    def setState(self, state, phase1Info={}):
        errorMessage = None
        try:
            msg = create_msg('setstate.' + state, body=phase1Info)
            self.front_req.send_json(msg)
            reply = self.front_req.recv_json()
        except zmq.Again:
            errorMessage = 'setState() timeout (%.1f sec)' % (self.timeout / 1000.)
            logging.info('setState() reinitializing zmq socket')
            self.front_req_init()
        except Exception as ex:
            errorMessage = 'setState() Exception: %s' % ex
        else:
            try:
                errorMessage = reply['body']['err_info']
            except KeyError:
                pass

        return errorMessage

    #
    # DaqControl.setConfig - set BEAM/NOBEAM
    #
    def setConfig(self, config):
        errorMessage = None
        try:
            msg = create_msg('setconfig.' + config)
            self.front_req.send_json(msg)
            reply = self.front_req.recv_json()
        except Exception as ex:
            errorMessage = 'setConfig() Exception: %s' % ex
        else:
            try:
                errorMessage = reply['body']['err_info']
            except KeyError:
                pass

        return errorMessage

    #
    # DaqControl.setRecord - set record flag
    #   True or False
    #
    def setRecord(self, recordIn):
        errorMessage = None
        if type(recordIn) == type(True):
            if recordIn:
                record = '1'
            else:
                record = '0'

            try:
                msg = create_msg('setrecord.' + record)
                self.front_req.send_json(msg)
                reply = self.front_req.recv_json()
            except Exception as ex:
                errorMessage = 'setRecord() Exception: %s' % ex
            else:
                try:
                    errorMessage = reply['body']['err_info']
                except KeyError:
                    pass
        else:
            errorMessage = 'setRecord() requires True or False'

        return errorMessage

    #
    # DaqControl.setBypass - set bypass_activedet flag
    #   True or False
    #
    def setBypass(self, bypassIn):
        errorMessage = None
        if type(bypassIn) == type(True):
            if bypassIn:
                bypass = '1'
            else:
                bypass = '0'

            try:
                msg = create_msg('setbypass.' + bypass)
                self.front_req.send_json(msg)
                reply = self.front_req.recv_json()
            except Exception as ex:
                errorMessage = 'setBypass() Exception: %s' % ex
            else:
                try:
                    errorMessage = reply['body']['err_info']
                except KeyError:
                    pass
        else:
            errorMessage = 'setBypass() requires True or False'

        return errorMessage

    #
    # DaqControl.setTransition - trigger a transition
    # The optional second argument is a dictionary containing
    # information that will be put into the phase1-json of the transition.
    # An example:
    # {'myvalue1':3 , 'myvalue2': {'myvalue3':72}}
    #
    def setTransition(self, transition, phase1Info={}):
        errorMessage = None
        try:
            msg = create_msg(transition, body=phase1Info)
            self.front_req.send_json(msg)
            reply = self.front_req.recv_json()
        except Exception as ex:
            errorMessage = 'setTransition() Exception: %s' % ex
        else:
            try:
                errorMessage = reply['body']['err_info']
            except KeyError:
                pass

        return errorMessage

    #
    # DaqControl.front_req_init - (re)initialize the front_req zmq socket
    #
    def front_req_init(self):
        # if socket previouly created, close it
        if self.front_req is not None:
            self.front_req.close()
        # create new socket
        self.front_req = self.context.socket(zmq.REQ)
        self.front_req.linger = 0
        self.front_req.RCVTIMEO = self.timeout
        self.front_req.connect(self.front_req_endpoint)

next_dict = {
    'reset' :       { 'unallocated' : 'rollcall',
                      'allocated' :   'rollcall',
                      'connected' :   'rollcall',
                      'configured' :  'rollcall',
                      'starting' :    'rollcall',
                      'paused' :      'rollcall',
                      'running' :     'rollcall' },

    'unallocated' : { 'reset' :       'reset',
                      'allocated' :   'alloc',
                      'connected' :   'alloc',
                      'configured' :  'alloc',
                      'starting' :    'alloc',
                      'paused' :      'alloc',
                      'running' :     'alloc' },

    'allocated' :   { 'reset' :       'reset',
                      'unallocated' : 'dealloc',
                      'connected' :   'connect',
                      'configured' :  'connect',
                      'starting' :    'connect',
                      'paused' :      'connect',
                      'running' :     'connect' },

    'connected' :   { 'reset' :       'reset',
                      'unallocated' : 'disconnect',
                      'allocated' :   'disconnect',
                      'configured' :  'configure',
                      'starting' :    'configure',
                      'paused' :      'configure',
                      'running' :     'configure' },

    'configured' :  { 'reset' :       'reset',
                      'unallocated' : 'unconfigure',
                      'allocated' :   'unconfigure',
                      'connected' :   'unconfigure',
                      'starting' :    'beginrun',
                      'paused' :      'beginrun',
                      'running' :     'beginrun' },

    'starting' :    { 'reset' :       'reset',
                      'unallocated' : 'endrun',
                      'allocated' :   'endrun',
                      'connected' :   'endrun',
                      'configured' :  'endrun',
                      'paused' :      'beginstep',
                      'running' :     'beginstep' },

    'paused' :      { 'reset' :       'reset',
                      'unallocated' : 'endstep',
                      'allocated' :   'endstep',
                      'connected' :   'endstep',
                      'configured' :  'endstep',
                      'starting' :    'endstep',
                      'running' :     'enable' },

    'running' :     { 'reset' :       'reset',
                      'unallocated' : 'disable',
                      'allocated' :   'disable',
                      'connected' :   'disable',
                      'configured' :  'disable',
                      'starting' :    'disable',
                      'paused' :      'disable' }
}


class ConfigurationScan:
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
        self.daqState = daqState
        self.args = args
        self.daqState_cv = Condition()
        self.stepDone_cv = Condition()
        self.stepDone = 0
        self.comm_thread.start()
        self.mon_thread.start()
        self.verbose = args.v
        self.pv_base = args.B
        self.motors = []                # set in configure()
        self._step_count = 0
        self.cydgram = dc.CyDgram()

        if args.g is None:
            self.groupMask = 1 << args.p
        else:
            self.groupMask = args.g

        # StepEnd is a cumulative count
        self.readoutCount = args.c
        self.readoutCumulative = 0

        # instantiate DaqPVA object
        self.pva = DaqPVA(platform=args.p, xpm_master=args.x, pv_base=args.B)

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

                # set EPICS PVs.
                # StepEnd is a cumulative count.
                self.readoutCumulative += self.readoutCount
                self.pva.pv_put(self.pva.pvStepEnd, self.readoutCumulative)
                self.pva.step_groups(mask=self.groupMask)
                self.pva.pv_put(self.pva.pvStepDone, 0)
                with self.stepDone_cv:
                    self.stepDone = 0
                    self.stepDone_cv.notify()

                # set DAQ state
                if phase1 is None:
                    errMsg = self.control.setState(state)
                else:
                    errMsg = self.control.setState(state, oldjson.loads(phase1))
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

            elif state=='shutdown':
                break

    def daq_monitor_thread(self):
        logging.debug('*** daq_monitor_thread')
        while True:
            part1, part2, part3, part4, part5, part6, part7, part8 = self.control.monitorStatus()
            if part1 is None:
                break
            elif part1 not in DaqDefs.transitions:
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

