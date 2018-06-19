"""
=====================================================================
ControlMsg - Control level message class

Author: Chris Ford <caf@slac.stanford.edu>

Based on kvmsg class by Min RK <benjaminrk@gmail.com>

"""

import sys
import zmq
from KVMsg import KVMsg

class ControlMsg(KVMsg):

    PORTBASE = 29960

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

    @classmethod
    def router_port(cls, platform):
        assert platform >= 0 and platform <= 7
        return cls.PORTBASE + platform

    @classmethod
    def pull_port(cls, platform):
        assert platform >= 0 and platform <= 7
        return 5559 # FIXME


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

    kvmap = {}
    # Test send and receive of simple message
    controlmsg = ControlMsg(1)
    controlmsg.key = b"key"
    controlmsg.body = b"body"
    if verbose:
        controlmsg.dump()
    controlmsg.send(output)
    controlmsg.store(kvmap)

    controlmsg2 = ControlMsg.recv(input)
    if verbose:
        controlmsg2.dump()
    assert controlmsg2.key == b"key"
    controlmsg2.store(kvmap)

    assert len(kvmap) == 1 # shouldn't be different

    # test send/recv with properties:
    controlmsg = ControlMsg(2, key=b"key", body=b"body")
    controlmsg[b"prop1"] = b"value1"
    controlmsg[b"prop2"] = b"value2"
    controlmsg[b"prop3"] = b"value3"
    assert controlmsg[b"prop1"] == b"value1"
    if verbose:
        controlmsg.dump()
    controlmsg.send(output)
    controlmsg2 = ControlMsg.recv(input)
    if verbose:
        controlmsg2.dump()
    # ensure properties were preserved
    assert controlmsg2.key == controlmsg.key
    assert controlmsg2.body == controlmsg.body
    assert controlmsg2.properties == controlmsg.properties
    assert controlmsg2[b"prop2"] == controlmsg[b"prop2"]

    print("OK")

if __name__ == '__main__':
    test_controlmsg('-v' in sys.argv)
