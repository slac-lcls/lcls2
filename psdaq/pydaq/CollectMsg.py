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

def test_collectmsg (verbose):
    print(" * collectmsg: ", end='')

    # Prepare our context and sockets
    ctx = zmq.Context()
    output = ctx.socket(zmq.DEALER)
    output.bind("ipc://collectmsg_selftest.ipc")
    input = ctx.socket(zmq.DEALER)
    input.connect("ipc://collectmsg_selftest.ipc")

    # Test send and receive of simple message
    collectmsg = CollectMsg(1)
    collectmsg.key = b"key"
    collectmsg.body = b"body"
    if verbose:
        collectmsg.dump()
    collectmsg.send(output)

    collectmsg2 = CollectMsg.recv(input)
    if verbose:
        collectmsg2.dump()
    assert collectmsg2.key == b"key"

    print("OK")

if __name__ == '__main__':
    test_collectmsg('-v' in sys.argv)
