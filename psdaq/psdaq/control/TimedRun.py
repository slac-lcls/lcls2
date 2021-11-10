import logging
import zmq
from threading import Thread, Event, Condition
from psdaq.control.ControlDef import ControlDef, front_pub_port, front_rep_port, create_msg
import time

class TimedRun:
    def __init__(self, control, *, daqState, args):
        self.control = control
        self.name = 'mydaq'
        self.context = zmq.Context()
        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.bind('inproc://timed_run')
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.connect('inproc://timed_run')
        self.comm_thread = Thread(target=self.daq_communicator_thread, args=(), daemon=True)
        self.mon_thread = Thread(target=self.daq_monitor_thread, args=(), daemon=True)
        self.ready = Event()
        self.daqState = daqState
        self.args = args
        self.daqState_cv = Condition()
        self.comm_thread.start()
        self.mon_thread.start()
        self.verbose = args.v

    # this thread tells the daq to go to a state and waits for the completion
    def daq_communicator_thread(self):
        logging.debug('*** daq_communicator_thread')
        while True:
            sss = self.pull_socket.recv().decode("utf-8")
            if ',' in sss:
                state, phase1 = sss.split(',', maxsplit=1)
            else:
                state, phase1 = sss, None

            logging.debug('*** received %s' % state)
            if state in ControlDef.states:
                # send 'daqstate(state)' and wait for complete
                errMsg = self.control.setState(state)
                if errMsg is not None:
                    logging.error('%s' % errMsg)
                    continue

                with self.daqState_cv:
                    while self.daqState != state:
                        logging.debug('daqState \'%s\', waiting for \'%s\'...' % (self.daqState, state))
                        self.daqState_cv.wait(1.0)
                        # check for shutdown with nonblocking read
                        try:
                            ttt = self.pull_socket.recv(flags=zmq.NOBLOCK).decode("utf-8")
                        except Exception as ex:
                            pass
                        else:
                            if ttt=='shutdown':
                                return

                    logging.debug('daqState \'%s\'' % self.daqState)

                if self.daqState == state:
                    self.ready.set()

            elif state=='shutdown':
                break

            else:
                logging.error(f'daq_communicator_thread unrecognized input: \'{state}\'')

    def daq_monitor_thread(self):
        logging.debug('*** daq_monitor_thread')
        while True:
            part1, part2, part3, part4, part5, part6, part7, part8 = self.control.monitorStatus()
            if part1 is None:
                break
            elif part1 == 'error':
                logging.error(f"{part2}")
            elif part1 == 'warning':
                logging.warning(f"{part2}")
            elif part1 not in ControlDef.transitions:
                continue

            # part1=transition, part2=state, part3=config
            with self.daqState_cv:
                self.daqState = part2
                self.daqState_cv.notify()

    def sleep(self, secs):
        logging.debug(f'begin {secs} second wait')
        time.sleep(secs)
        logging.debug(f'end {secs} second wait')

    def set_connected_state(self):
        self.push_socket.send_string('connected')
        # wait for complete
        self.ready.wait()
        self.ready.clear()

    def set_running_state(self):
        self.push_socket.send_string('running')
        # wait for complete
        self.ready.wait()
        self.ready.clear()

    def stage(self):
        # done once at start of scan
        # put the daq into the right state ('connected')
        self.set_connected_state()

    def unstage(self):
        # done once at end of scan
        # put the daq into the right state ('connected')
        self.set_connected_state()

