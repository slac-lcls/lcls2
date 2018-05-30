"""
=====================================================================
CMMsg - Collection Manager message class

Author: Chris Ford <caf@slac.stanford.edu>

Based on kvmsg class by Min RK <benjaminrk@gmail.com>

"""

import sys
import zmq
from kvmsg import KVMsg

class CMMsg(KVMsg):

    PORTBASE = 29980

    # message keys
    STARTPING       = b'STARTPING'
    PING            = b'PING'
    PONG            = b'PONG'
    HELLO           = b'HELLO'
    HELLODRP        = b'HELLODRP'
    PORTS           = b'PORTS'
    PORTSDRP        = b'PORTSDRP'
    STARTPLAT       = b'STARTPLAT'
    PLATSTARTED     = b'PLATSTARTED'
    PLAT            = b'PLAT'
    STARTALLOC      = b'STARTALLOC'
    ALLOCSTARTED    = b'ALLOCSTARTED'
    ALLOC           = b'ALLOC'
    STARTCONNECT    = b'STARTCONNECT'
    CONNECTSTARTED  = b'CONNECTSTARTED'
    CONNECT         = b'CONNECT'
    CONNECTED       = b'CONNECTED'
    STARTKILL       = b'STARTKILL'
    KILLSTARTED     = b'KILLSTARTED'
    KILL            = b'KILL'
    STARTDIE        = b'STARTDIE'
    DIESTARTED      = b'DIESTARTED'
    DIE             = b'DIE'
    STARTEXP        = b'STARTEXP'
    STARTDUMP       = b'STARTDUMP'
    DUMPSTARTED     = b'DUMPSTARTED'
    GETSTATE        = b'GETSTATE'
    STATE           = b'STATE'
    HUH             = b'HUH?'

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
    cmmsg = CMMsg(1)
    cmmsg.key = b"key"
    cmmsg.body = b"body"
    if verbose:
        cmmsg.dump()
    cmmsg.send(output)
    cmmsg.store(kvmap)

    cmmsg2 = CMMsg.recv(input)
    if verbose:
        cmmsg2.dump()
    assert cmmsg2.key == b"key"
    cmmsg2.store(kvmap)

    assert len(kvmap) == 1 # shouldn't be different

    # test send/recv with properties:
    cmmsg = CMMsg(2, key=b"key", body=b"body")
    cmmsg[b"prop1"] = b"value1"
    cmmsg[b"prop2"] = b"value2"
    cmmsg[b"prop3"] = b"value3"
    assert cmmsg[b"prop1"] == b"value1"
    if verbose:
        cmmsg.dump()
    cmmsg.send(output)
    cmmsg2 = CMMsg.recv(input)
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
