"""
=====================================================================
ControlMsg - Control level message class

Author: Chris Ford <caf@slac.stanford.edu>

Based on kvmsg class by Min RK <benjaminrk@gmail.com>

"""

import sys
import zmq
from JsonMsg import JsonMsg

class ControlMsg(JsonMsg):

    # message keys
    STARTPING   = b'STARTPING'
    PING        = b'PING'
    PONG        = b'PONG'
    HELLO       = b'HELLO'
    PORTS       = b'PORTS'
    STARTEXP    = b'STARTEXP'
    STARTDUMP   = b'STARTDUMP'
    DUMPSTARTED = b'DUMPSTARTED'
    GETSTATE    = b'GETSTATE'
    STATE       = b'STATE'
    HUH         = b'HUH?'


# ---------------------------------------------------------------------
# Runs self test of class

def test_controlmsg (verbose):
    print(" * controlmsg: ", end='')

    # Prepare our context and sockets
    ctx = zmq.Context()
    output = ctx.socket(zmq.DEALER)
    output.bind("ipc://controlmsg_selftest.ipc")
    input = ctx.socket(zmq.DEALER)
    input.connect("ipc://controlmsg_selftest.ipc")

    # Test send and receive of simple message
    controlmsg = ControlMsg(1)
    controlmsg.key = b"key"
    controlmsg.body = b"body"
    if verbose:
        controlmsg.dump()
    controlmsg.send(output)

    controlmsg2 = ControlMsg.recv(input)
    if verbose:
        controlmsg2.dump()
    assert controlmsg2.key == b"key"

    print("OK")

if __name__ == '__main__':
    test_controlmsg('-v' in sys.argv)
