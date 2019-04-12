"""
Sends in loop ZMQ messages with timestamp to specified socket.

Usage::
    python psdaq/psdaq/control_gui/test_QWZMQListener.py   # on platform 6 by default
    python psdaq/psdaq/control_gui/test_QWZMQListener.py 7 # on platform 7
"""

import sys
import zmq
from time import time, strftime, localtime, sleep
from psdaq.control.collection import front_pub_port

#----------

PLATFORM = 6 if len(sys.argv) < 2 else int(sys.argv[1])
PORT = front_pub_port(PLATFORM) # 30016
URI = 'tcp://*:%d' % PORT # 'tcp://*:30016
SLEEP_SEC = 1
TOPIC = b'' # b'10001'

print('URI: %s' % URI)

context = zmq.Context(1)
socket = context.socket(zmq.PUB)
socket.bind(URI)

#----------

if __name__ == '__main__':

    while True :
        t0_sec = time()
        tstamp = strftime('%Y%m%dT%H%M%S', localtime(t0_sec))
        s = 'time stamp ' + tstamp
        print('send ZMQ msg: %s' % s)
        #socket.send_string(s)
        socket.send_multipart([TOPIC, bytes(s.encode('utf-8'))])
        #socket.send(bytes(("10001 %s" % s).encode('utf-8')))
        sleep(SLEEP_SEC) # sec

#----------
