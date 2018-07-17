"""
=====================================================================
CollectMsg - Collection Manager message class

Author: Chris Ford <caf@slac.stanford.edu>

Based on kvmsg class by Min RK <benjaminrk@gmail.com>

"""

import sys
import zmq
from JsonMsg import JsonMsg

class CollectMsg(JsonMsg):

    PORTBASE = 29980

    # message keys
    STARTPING       = b'STARTPING'
    PING            = b'PING'
    PONG            = b'PONG'
    HELLO           = b'HELLO'
    CONNECTINFO     = b'CONNECTINFO'
    RESET           = b'RESET'
    RESETSTARTED    = b'RESETSTARTED'
    RESETFAILED     = b'RESETFAILED'
    NOPLAT          = b'NOPLAT'
    PLAT            = b'PLAT'
    PLATSTARTED     = b'PLATSTARTED'
    PLATFAILED      = b'PLATFAILED'
    ALLOC           = b'ALLOC'
    ALLOCSTARTED    = b'ALLOCSTARTED'
    ALLOCFAILED     = b'ALLOCFAILED'
    CONNECT         = b'CONNECT'
    CONNECTSTARTED  = b'CONNECTSTARTED'
    CONNECTFAILED   = b'CONNECTFAILED'
    KILL            = b'KILL'
    KILLSTARTED     = b'KILLSTARTED'
    DIE             = b'DIE'
    DIESTARTED      = b'DIESTARTED'
    DUMP            = b'DUMP'
    DUMPSTARTED     = b'DUMPSTARTED'
    GETSTATE        = b'GETSTATE'
    STATE           = b'STATE'
    HUH             = b'HUH?'
    OK              = b'OK'
    ERROR           = b'ERROR'

    @classmethod
    def router_port(cls, platform):
        assert platform >= 0 and platform <= 7
        return cls.PORTBASE + platform

    @classmethod
    def pub_port(cls, platform):
        assert platform >= 0 and platform <= 7
        return cls.PORTBASE + platform + 10


# ---------------------------------------------------------------------
# Runs self test of class

def test_cmmsg (verbose):
    print(" * cmmsg: ", end='')

    # Prepare our context and sockets
    ctx = zmq.Context()
    output = ctx.socket(zmq.DEALER)
    output.bind("ipc://cmmsg_selftest.ipc")
    input = ctx.socket(zmq.DEALER)
    input.connect("ipc://cmmsg_selftest.ipc")

    kvmap = {}
    # Test send and receive of simple message
    cmmsg = CollectMsg(1)
    cmmsg.key = b"key"
    cmmsg.body = b"body"
    if verbose:
        cmmsg.dump()
    cmmsg.send(output)
    cmmsg.store(kvmap)

    cmmsg2 = CollectMsg.recv(input)
    if verbose:
        cmmsg2.dump()
    assert cmmsg2.key == b"key"
    cmmsg2.store(kvmap)

    assert len(kvmap) == 1 # shouldn't be different

    # test send/recv with properties:
    cmmsg = CollectMsg(2, key=b"key", body=b"body")
    cmmsg[b"prop1"] = b"value1"
    cmmsg[b"prop2"] = b"value2"
    cmmsg[b"prop3"] = b"value3"
    assert cmmsg[b"prop1"] == b"value1"
    if verbose:
        cmmsg.dump()
    cmmsg.send(output)
    cmmsg2 = CollectMsg.recv(input)
    if verbose:
        cmmsg2.dump()
    # ensure properties were preserved
    assert cmmsg2.key == cmmsg.key
    assert cmmsg2.body == cmmsg.body
    assert cmmsg2.properties == cmmsg.properties
    assert cmmsg2[b"prop2"] == cmmsg[b"prop2"]

    print("OK")

if __name__ == '__main__':
    test_cmmsg('-v' in sys.argv)
