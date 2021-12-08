import sys
import socket
import argparse
from p4p.client.thread import Context
from threading import Thread, Event, Condition, Timer
import logging
import time

class TranControl(object):

    def __init__(self,args):
        print('__init__')

        self.args = args
        self.pv_xpm_base      = 'DAQ:LAB2:XPM:%d'         % args.x
        self.pvListMsgHeader  = []  # filled in at alloc
        for g in range(8):
            if self.args.p & (1 << g):
                self.pvListMsgHeader.append(self.pv_xpm_base+":PART:"+str(g)+':MsgHeader')


        self.pvGroupMsgInsert = self.pv_xpm_base+':GroupMsgInsert'
        self.ctxt = Context('pva', nt=None)

    def pv_put(self, pvName, val):

        retval = False

        try:
            self.ctxt.put(pvName, val)
        except TimeoutError:
            self.report_error("self.ctxt.put('%s', %d) timed out" % (pvName, val))
        except Exception:
            self.report_error("self.ctxt.put('%s', %d) failed" % (pvName, val))
        else:
            retval = True
            logging.debug("self.ctxt.put('%s', %d)" % (pvName, val))

        return retval

    def run(self):
        print('run')
        self.groups = self.args.p
        self.pv_put(self.pvGroupMsgInsert  ,self.args.p)

        timer = 1./self.args.r

        def expired():
            self.pv_put(self.pvGroupMsgInsert, self.args.p)
            self.transitions = Timer(timer,expired)
            self.transitions.start()

        self.transitions = Timer(timer,expired)
        self.transitions.start()

        while(True):
            time.sleep(1)

def main():
    print('main')
    parser = argparse.ArgumentParser(description='xpm scan test')
    parser.add_argument('-x', metavar='XPM', type=int, default=2,
                        help='master XPM')
    parser.add_argument('-p', metavar='PARTMASK',type=int, default=1,
                        help='partition mask (default 1)')
    parser.add_argument('-r', metavar='RATE', type=float, default=1., help='rate,Hz')
    parser.add_argument('-t', metavar='TRANSITION', type=int, default=10, help='transitionId')

    args = parser.parse_args()
    
#    logging.basicConfig(level=logging.DEBUG)

    c = TranControl(args)
    c.run()

if __name__ == '__main__':
    main()
