"""
Sends in loop ZMQ messages with timestamp to specified socket.

"""

SLEEP_SEC = 1
TOPIC = b"10001"

import zmq
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind('tcp://*:5556')

if __name__ == '__main__':

    from time import time, strftime, localtime, sleep

    while True :
        t0_sec = time()
        tstamp = strftime('%Y%m%dT%H%M%S', localtime(t0_sec))
        s = 'timestamp: ' + tstamp
        print('send ZMQ msg: %s' %s)
        socket.send_multipart([TOPIC, bytes(s.encode('utf-8'))])
        #socket.send_string(s)
        #socket.send(bytes(("10001 %s" % s).encode('utf-8')))
        sleep(SLEEP_SEC) # sec
