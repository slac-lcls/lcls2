#!/usr/bin/env python

"""
repserver - ZMQ REP socket server

This is an example server processing multipart ZMQ messages.

The 'Hello' command simply returns 'World' reply.

The 'Load' command returns a 'LOADED' reply plus a stored value
in the 2nd part of the 'LOADED' message.

The 'Store' command returns a 'STORED' reply, and it stores
the value received from the 2nd part of the 'Store' message.

Author: Chris Ford <caf@slac.stanford.edu>
"""

import zmq
import sys
import time
import logging
import pprint
import argparse
import struct
from psana.dgrammanager import DgramManager
from psana.dgrammanager import setnames
from psana import dgram


def main():

    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', action='store_true', help='be verbose')
    args = parser.parse_args()

    if args.v:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info('repserver starting')

    # context and sockets
    ctx = zmq.Context()
    repSocket = ctx.socket(zmq.REP)
    repSocket.bind("tcp://*:5560")

    sequence = 0 
    storedDgram = None
    try:
        while True:

            # Wait for next request from client
            msg = repSocket.recv_multipart()


            print("Received request: %s" % msg[0])
            time.sleep(0.5)

            # Send reply back to client
            if len(msg) == 0:
                repSocket.send(b"Huh?")
            elif msg[0] == b"Hello":
                repSocket.send(b"World")
            elif msg[0] == b"Load":
                # doLoad(repSocket)
                if storedDgram:
                    xtc_len = struct.pack("<q", len(storedDgram))
                    repSocket.send_multipart([b"LOADED", storedDgram[:52], storedDgram])
                    storedDgram = None
                else:
                    repSocket.send_multipart([b"No data found"])


            elif msg[0] == b"Store":

                config = dgram.Dgram(view = msg[1])
                setnames(config)
                if config:
                    print("Python server parsed the datagram")
                    # Make changes to the datagram with cydgram
                    # Get bytes object back to send to c++ code

                storedDgram = msg[1]

                repSocket.send(b"STORED")
            else:
                repSocket.send(b"Huh?")

    except KeyboardInterrupt:
        logging.debug("Interrupt received")

    # Clean up
    logging.debug("Clean up")

    # close zmq sockets
    repSocket.close()

    # terminate zmq context
    ctx.term()

    logging.info('repserver exiting')

if __name__ == '__main__':
    main()
