"""
=====================================================================
CMMsg - Collection Manager message class

Author: Chris Ford <caf@slac.stanford.edu>

Based on kvmsg class by Min RK <benjaminrk@gmail.com>

"""

import struct # for packing integers
import sys
import os
from uuid import uuid4

import zmq
import pickle

class CMMsg(object):

    PORTBASE = 29980

    # message keys
    STARTPING   = b'STARTPING'
    PING        = b'PING'
    PONG        = b'PONG'
    HELLO       = b'HELLO'
    PORTS       = b'PORTS'
    STARTPH1    = b'STARTPH1'
    PH1STARTED  = b'PH1STARTED'
    PH1         = b'PH1'
    STARTPH2    = b'STARTPH2'
    PH2STARTED  = b'PH2STARTED'
    PH2         = b'PH2'
    STARTKILL   = b'STARTKILL'
    KILLSTARTED = b'KILLSTARTED'
    KILL        = b'KILL'
    STARTDIE    = b'STARTDIE'
    DIESTARTED  = b'DIESTARTED'
    DIE         = b'DIE'
    STARTEXP    = b'STARTEXP'
    STARTDUMP   = b'STARTDUMP'
    DUMPSTARTED = b'DUMPSTARTED'
    GETSTATE    = b'GETSTATE'
    STATE       = b'STATE'
    HUH         = b'HUH?'

    """
    Message is formatted on wire as 4 frames:
    frame 0: key (0MQ string)
    frame 1: sequence (8 bytes, network order)
    frame 2: uuid (blob, 16 bytes)
    frame 3: properties (pickled Python dictionary)
    """
    key = None
    sequence = 0
    uuid=None
    properties = None

    @staticmethod
    def encode_properties(properties_dict):
        prop_s = pickle.dumps([])
        try:
            prop_s = pickle.dumps(properties_dict)
        except Exception as ex:
            print("encode_properties(): %s" % ex)
        return prop_s

    @staticmethod
    def decode_properties(prop_s):
        prop = {}
        try:
            prop = pickle.loads(prop_s)
        except Exception as ex:
            print("decode_properties(): %s" % ex)
        return prop

    def __init__(self, *, sequence=0, uuid=None, key=None, properties=None):
        assert isinstance(sequence, int)
        self.sequence = sequence
        if uuid is None:
            uuid = uuid4().bytes
        self.uuid = uuid
        self.key = key
        self.properties = {} if properties is None else properties

    # dictionary access maps to properties:
    def __getitem__(self, k):
        return self.properties[k]

    def __setitem__(self, k, v):
        self.properties[k] = v

    def get(self, k, default=None):
        return self.properties.get(k, default)

    def store(self, dikt):
        """Store me in a dict if I have anything to store 
        else delete me from the dict."""
        if self.key is not None:
            dikt[self.key] = self
        elif self.key in dikt:
            del dikt[self.key]

    def send(self, socket):
        """Send message to socket; any empty frames are sent as such."""
        key = b'' if self.key is None else self.key
        seq_s = struct.pack('!q', self.sequence)
        prop_s = self.encode_properties(self.properties)
        socket.send_multipart([ key, seq_s, self.uuid, prop_s ])

    @classmethod
    def recv(cls, socket):
        """Reads message from socket, returns new cmmsg instance."""
        return cls.from_msg(socket.recv_multipart())

    @classmethod
    def from_msg(cls, msg):
        """Construct message from a multipart message"""
        key, seq_s, uuid, prop_s = msg
        key = key if key else None
        seq = struct.unpack('!q',seq_s)[0]
        prop = cls.decode_properties(prop_s)
        return cls(sequence=seq, uuid=uuid, key=key, properties=prop)
    
    def __repr__(self):
        mstr = "[seq:{seq}][key:{key}][props:{props}]".format(
            seq=self.sequence,
            # uuid=hexlify(self.uuid),
            key=self.key,
            props=self.encode_properties(self.properties),
        )
        return mstr

    @classmethod
    def router_port(cls, platform):
        assert platform >= 0 and platform <= 7
        return cls.PORTBASE + platform

    @classmethod
    def pub_port(cls, platform):
        assert platform >= 0 and platform <= 7
        return cls.PORTBASE + platform + 10

    def dump(self):
        print("<<", str(self), ">>", file=sys.stderr)

# ---------------------------------------------------------------------
# Runs self test of class

def test_cmmsg (verbose=True):
    print(" * cmmsg: ", end='')

    # Prepare our context and sockets
    ctx = zmq.Context()
    output = ctx.socket(zmq.DEALER)
    output.bind("ipc://cmmsg_selftest.ipc")
    input = ctx.socket(zmq.DEALER)
    input.connect("ipc://cmmsg_selftest.ipc")

    cmmap = {}
    # Test send and receive of simple message
    cmmsg = CMMsg(sequence=1)
    cmmsg.key = b"key"
    if verbose:
        cmmsg.dump()
    cmmsg.send(output)
    cmmsg.store(cmmap)

    cmmsg2 = CMMsg.recv(input)
    if verbose:
        cmmsg2.dump()
    assert cmmsg2.key == b"key"
    cmmsg2.store(cmmap)

    assert len(cmmap) == 1 # shouldn't be different

    # test send/recv with properties:
    cmmsg = CMMsg(sequence=2, key=b'key')
    cmmsg['level'] = 0
    cmmsg['pid'] = 25707
    cmmsg['ip'] = '172.21.21.35'
    assert cmmsg["level"] == 0
    if verbose:
        cmmsg.dump()
    cmmsg.send(output)
    cmmsg2 = CMMsg.recv(input)
    if verbose:
        cmmsg2.dump()
    # ensure properties were preserved
    assert cmmsg2.key == cmmsg.key
    assert cmmsg2.properties == cmmsg.properties
    assert cmmsg2["pid"] == cmmsg["pid"]

    print("OK")

if __name__ == '__main__':
    test_cmmsg('-v' in sys.argv)
