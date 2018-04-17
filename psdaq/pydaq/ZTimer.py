"""
=====================================================================
ztimer - zmq based timer class

Author: Chris Ford <caf@slac.stanford.edu>

"""

import threading
import time
import zmq
import logging

class ZTimer (threading.Thread):

    def __init__(self, name, ctx, endpoint, timerList):
        threading.Thread.__init__(self)
        self.name = name
        self.ctx = ctx
        self.endpoint = endpoint
        self.timerList = timerList
        self._count = 0
        self._rollover = 1000

    def run(self):
        logging.debug("Starting " + self.name)
        self.send_timer_events()
        logging.debug("Exiting " + self.name)

    def increment(self):
        self._count = (self._count + 1) % self._rollover

    def count(self, offset=0):
        return (self._count - offset) % self._rollover

    def send_timer_events(self):
        # create, set timeout, and connect to socket
        self.sendTimer = self.ctx.socket(zmq.PAIR)
        self.sendTimer.rcvtimeo = 1000
        self.sendTimer.connect(self.endpoint)

        while True:
            try:
                # receive with 1 sec timeout
                msg = self.sendTimer.recv()
            except zmq.Again as e:
                # tick
                logging.debug("tick %d" % self.count())
                pass
            else:
                break

            # send message(s) based on timer count
            for tt in self.timerList:
                if (self.count(tt['offset']) % tt['period']) == 0:
                    self.sendTimer.send(tt['msg'])
                    logging.debug("Sent <%s> from timer" % tt['msg'].decode())

            # increment timer count
            self.increment()

        # clean up
        logging.debug(self.name + " clean up")
        self.sendTimer.close()

###############################################################################################

def test_ztimer(verbose = True):

    if verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    timers = [ { 'period' : 10, 'offset' :  0, 'msg' : b'PERIOD10_OFFSET0'  },
               { 'period' : 10, 'offset' :  1, 'msg' : b'PERIOD10_OFFSET1'  },
               { 'period' : 30, 'offset' :  0, 'msg' : b'PERIOD30_OFFSET0'} ]

    # Create zmq context
    context = zmq.Context()

    # Bind to inproc: endpoint
    endpoint1 = "inproc://timer1"
    receiveTimer1 = context.socket(zmq.PAIR)
    receiveTimer1.sndtimeo = 0
    receiveTimer1.bind(endpoint1)

    try:
        # Create timer thread
        timer1 = ZTimer("Timer-1", context, endpoint1, timers)
        timer1.start()

        # Wait for timer messages
        while True:
            msg = receiveTimer1.recv()
            logging.info("Message received: <%s>" % msg)

    except KeyboardInterrupt:
        logging.debug("Interrupt received")

    # Clean up
    logging.debug("Clean up")
    try:
        receiveTimer1.send(b"")   # signal timer 1 to exit
    except zmq.Again:
        pass

    time.sleep(.25)
    receiveTimer1.close() # close zmq socket
    context.term()        # terminate zmq context
    timer1.join()         # join timer thread

    logging.debug("Exiting Main Thread")
