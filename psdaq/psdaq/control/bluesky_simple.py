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
import threading
import zmq
import asyncio
import time

from psdaq.control.control import DaqControl
import argparse

class MyDAQ:
    def __init__(self, control, motor):
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
        self.motor = motor
        self.daqState = 'noconnect'
        self.daqState_cv = threading.Condition()
        self.comm_thread.start()
        self.mon_thread.start()

    def read(self):
        # stuff we want to give back to user running bluesky
        # e.g. how many events we took.  called when trigger status
        # is done&&successful.
        # typically called by trigger_and_read()
        print('*** here in read')
        return {}
        return dict(('channel1',
             {'value': 5, 'timestamp': 1472493713.271991}),
             ('channel2',
             {'value': 16, 'timestamp': 1472493713.539238}))

    def describe(self):
        # stuff we want to give back to user running bluesky
        print('*** here in describe')
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
        print('*** daq_communicator_thread')
        while True:
            state = self.pull_socket.recv()
            print('*** received',state)
            if state==b'starting':
                # send 'daqstate(starting)' and wait for complete
                # we can block here since we are not in the bluesky
                # event loop
                errMsg = self.control.setState('starting')
                if errMsg is not None:
                    print('*** error:', errMsg)
                    continue
                with self.daqState_cv:
                    while self.daqState != 'starting':
                        print('daqState \'%s\', waiting for \'starting\'...' % self.daqState)
                        self.daqState_cv.wait(1.0)
                    print('daqState \'%s\'' % self.daqState)
                self.ready.set()
            elif state==b'running':
                # launch the step with 'daqstate(running)' (with the
                # scan values for the daq to record to xtc2).
                # normally should block on "complete" from the daq here.
                errMsg = self.control.setState('running',{'beginstep':{self.motor.name:self.motor.position}})
                if errMsg is not None:
                    print('*** error:', errMsg)
                    continue
                with self.daqState_cv:
                    while self.daqState != 'running':
                        print('daqState \'%s\', waiting for \'running\'...' % self.daqState)
                        self.daqState_cv.wait(1.0)
                    print('daqState \'%s\'' % self.daqState)
                # tell bluesky step is complete
                # this line is needed in ReadableDevice mode to flag completion
                self.status._finished(success=True)
            elif state==b'shutdown':
                break

    def daq_monitor_thread(self):
        print('*** daq_monitor_thread')
        while True:
            part1, part2, part3, part4 = self.control.monitorStatus()
            if part1 is None:
                break
            elif part1 == 'error':
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
        self.status.done = False
        self.status.success = False
        # tell the control level to do a step in the scan
        # to-do: pass it the motor positions, ideally both
        # requested/measured positions.
        # maybe don't launch the step directly here
        # with a daqstate command, since it would block
        # the event-loop?
        print('*** here in trigger',self.motor.position,self.motor.read())
        # this dict should be put into beginstep phase1 json
        motor_dict = {'motor1':self.motor.position,
                      'motor2':self.motor.position}
        self.push_socket.send_string('running')
        return self.status

    # for the FlyableDevice style, which we don't use for the DAQ
    # since we only get one kickoff call for all steps in the scan
    def kickoff(self):
        # do one step
        print('*** here in kickoff')
        self.status = Status()
        self.status.done = True
        self.status.success = True
        # tell the control level to start the scan
        self.push_socket.send_string('step')
        return self.status

    # for the FlyableDevice style
    def complete(self):
        print('*** here in complete')
        time.sleep(3)
        self.status = Status()
        self.status.done = True
        self.status.success = True
        return self.status

    # for the FlyableDevice style
    def collect(self):
        return {}

    # for the FlyableDevice style
    def describe_collect(self):
        return {}

    def read_configuration(self):
        # done at the first read after a configure
        return {}

    def describe_configuration(self):
        # the metadata for read_configuration()
        return {}

    def configure(self, *args, **kwargs):
        return (self.read_configuration(),self.read_configuration())

    def stage(self):
        # done once at start of scan
        # put the daq into the right state ('starting')
        print('*** here in stage')
        self.push_socket.send_string('starting')
        # wait for complete. is this a coroutine, so we shouldn't block?
        self.ready.wait()
        self.ready.clear()
        
        return [self]

    def unstage(self):
        return [self]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, choices=range(0, 8), default=0,
                        help='platform (default 0)')
    parser.add_argument('-C', metavar='COLLECT_HOST', default='localhost',
                        help='collection host (default localhost)')
    parser.add_argument('-t', type=int, metavar='TIMEOUT', default=10000,
                        help='timeout msec (default 10000)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--config', metavar='ALIAS', help='configuration alias')
    group.add_argument('-B', action="store_true", help='shortcut for --config BEAM')
    args = parser.parse_args()

    # instantiate DaqControl object
    control = DaqControl(host=args.C, platform=args.p, timeout=args.t)

    config = None
    if args.config:
        config = args.config
    elif args.B:
        config = "BEAM"

    if config:
        # config alias request
        rv = control.setConfig(config)
        if rv is not None:
            print('Error: %s' % rv)

    RE = RunEngine({})

    # cpo thinks this is more for printout of each step
    from bluesky.callbacks.best_effort import BestEffortCallback
    bec = BestEffortCallback()

    # Send all metadata/data captured to the BestEffortCallback.
    RE.subscribe(bec)

    # Make plots update live while scans run.
    #from bluesky.utils import install_kicker
    #install_kicker()

    from ophyd.sim import det, motor
    from bluesky.plans import scan, count
    from bluesky.preprocessors import fly_during_wrapper

    mydaq = MyDAQ(control,motor)
    dets = [mydaq]   # just one in this case, but it could be more than one

    print('motor',motor.position,motor.name) # in some cases we have to look at ".value"
    RE(scan(dets, motor, -1, 1, 3))
    print('motor',motor.position)
    #RE(count(dets, num=3))

    # only 1 callback! and 3 steps inside it.  doesn't feel useful for us
    #RE(fly_during_wrapper(count([det], num=3), dets))

    mydaq.push_socket.send_string('shutdown') #shutdown the daq thread
    mydaq.comm_thread.join()

if __name__ == '__main__':
    main()
