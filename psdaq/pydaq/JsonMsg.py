"""
=====================================================================
JsonMsg - JSON message class

Author: Chris Ford <caf@slac.stanford.edu>

"""

import sys

import zmq
# zmq.jsonapi ensures bytes, instead of unicode:
import zmq.utils.jsonapi as json

class JsonMsg(object):
    """
    Message is formatted on wire as 2 frames:
    frame 0: key (0MQ string)
    frame 1: body (blob)
    """
    key = None
    body = None

    def __init__(self, key=None, body=None):
        self.key = key
        self.body = body

    def send(self, socket):
        """Send key-value message to socket; any empty frames are sent as such."""
        key = b'' if self.key is None else self.key
        body = b'' if self.body is None else self.body
        socket.send_multipart([ key, body ])

    @classmethod
    def recv(cls, socket):
        """Reads key-value message from socket, returns new kvmsg instance."""
        return cls.from_msg(socket.recv_multipart())

    @classmethod
    def from_msg(cls, msg):
        """Construct key-value message from a multipart message"""
        key, body = msg
        key = key if key else None
        body = body if body else None
        return cls(key=key, body=body)
    
    def dump(self):
        if self.body is None:
            size = 0
            data='NULL'
        else:
            size = len(self.body)
            data=repr(self.body)
        print >> sys.stderr, "[key:{key}][size:{size}] {data}".format(
            key=self.key,
            size=size,
            data=data,
        )

# ---------------------------------------------------------------------
# Runs self test of class

def test_kvmsg (verbose):
    print(" * kvmsg: ", end='')

    # Prepare our context and sockets
    ctx = zmq.Context()
    output = ctx.socket(zmq.DEALER)
    output.bind("ipc://kvmsg_selftest.ipc")
    input = ctx.socket(zmq.DEALER)
    input.connect("ipc://kvmsg_selftest.ipc")

    kvmap = {}
    # Test send and receive of simple message
    kvmsg = JsonMsg(1)
    kvmsg.key = b"key"
    kvmsg.body = b"body"
    if verbose:
        kvmsg.dump()
    kvmsg.send(output)
    kvmsg.store(kvmap)

    kvmsg2 = JsonMsg.recv(input)
    if verbose:
        kvmsg2.dump()
    assert kvmsg2.key == b"key"
    kvmsg2.store(kvmap)

    assert len(kvmap) == 1 # shouldn't be different

    # test send/recv with properties:
    kvmsg = JsonMsg(2, key=b"key", body=b"body")
    kvmsg[b"prop1"] = b"value1"
    kvmsg[b"prop2"] = b"value2"
    kvmsg[b"prop3"] = b"value3"
    assert kvmsg[b"prop1"] == b"value1"
    if verbose:
        kvmsg.dump()
    kvmsg.send(output)
    kvmsg2 = JsonMsg.recv(input)
    if verbose:
        kvmsg2.dump()
    # ensure properties were preserved
    assert kvmsg2.key == kvmsg.key
    assert kvmsg2.body == kvmsg.body
    assert kvmsg2.properties == kvmsg.properties
    assert kvmsg2[b"prop2"] == kvmsg[b"prop2"]

    print("OK")

if __name__ == '__main__':
    test_kvmsg('-v' in sys.argv)
