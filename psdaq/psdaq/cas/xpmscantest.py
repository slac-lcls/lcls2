import sys
import socket
import argparse
from psdaq.control.control import DaqPVA
from threading import Thread, Event, Condition, Timer
import logging

class ScanControl(object):

    def __init__(self,args):

        self.args = args
        self.pva = DaqPVA(platform=args.p, xpm_master=args.x, pv_base='DAQ:LAB2')

        self.groups = int(1<<args.p)
        self.pva.pv_put(self.pva.pvStepGroups  ,self.groups)

        def expired():
            self.pva.pv_put(self.pva.pvGroupMsgInsert, self.groups)
            self.transitions = Timer(args.t,expired)
            self.transitions.start()

        self.transitions = Timer(args.t,expired)
        self.transitions.start()

        self.step_done = Event()

        def callback(done):
            logging.debug(f'callback {done}')
            if int(done):
                self.step_done.set()

        self.pva.monitor_StepDone(callback=callback)

    def run(self):
        self.pva.pv_put(self.pva.pvGroupL0Reset,self.groups)
        for i in range(self.args.s):
            logging.debug(f'begin step {i}')
            self.step_done.clear()
            self.pva.pv_put(self.pva.pvStepEnd , (i+1)*self.args.e)
            self.pva.pv_put(self.pva.pvStepDone, 0)
            self.pva.pv_put(self.pva.pvGroupL0Enable, self.groups)
            self.step_done.wait()
            logging.debug(f'end step {i}')

def main():

    parser = argparse.ArgumentParser(description='xpm scan test')
    parser.add_argument('-x', metavar='XPM', type=int, default=3,
                        help='master XPM')
    parser.add_argument('-p', metavar='PART',type=int, choices=range(0, 8), default=0,
                        help='partition (default 0)')
    parser.add_argument('-e', metavar='EVENTS', type=int, default=1000, help='event per step')
    parser.add_argument('-s', metavar='STEPS', type=int, default=1000, help='steps')
    parser.add_argument('-r', metavar='RATE', type=int, default=0, help='rate')
    parser.add_argument('-t', metavar='TIMER', type=float, default=1.0, help='timer')
    parser.add_argument('-n', metavar='NCYCLE', type=int, default=50, help='ncycles')

    args = parser.parse_args()
    
    logging.basicConfig(level=logging.DEBUG)

    c = ScanControl(args)
    for i in range(args.n):
        c.run()

if __name__ == '__main__':
    main()
