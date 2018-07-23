"""
=====================================================================
JsonMsg - JSON message class

Author: Chris Ford <caf@slac.stanford.edu>

"""

import sys
import logging

import zmq
# zmq.jsonapi ensures bytes, instead of unicode:
import zmq.utils.jsonapi as json

class JsonMsg(object):
    """
    Message is formatted on wire as 2 or 3 frames:
      frame 0: key (0MQ string)
      frame 1: body (blob)
    or
      frame 0: identity
      frame 1: key (0MQ string)
      frame 2: body (blob)
    """
    identity = None
    key = None
    body = {}
    json_body = json.dumps({})

    def __init__(self, *, identity=None, key=None, body={}):
        self.identity = identity
        self.key = key
        if isinstance(body, dict):
            self.body = body
            self.json_body = json.dumps(body)
        else:
            raise TypeError("message body must be of type dict")

    def send(self, socket):
        """Send message to socket."""
        key = b'' if self.key is None else self.key
        if self.identity is None:
            socket.send_multipart([ key, self.json_body ])
        else:
            socket.send_multipart([ self.identity, key, self.json_body ])

    @classmethod
    def recv(cls, socket):
        """Reads message from socket, returns new jsonmsg instance."""
        return cls.from_msg(socket.recv_multipart())

    @classmethod
    def from_msg(cls, msg):
        """Construct message from a multipart message"""
        if len(msg) == 2:
            identity = None
            key, json_body = msg
        elif len(msg) == 3:
            identity, key, json_body = msg
        else:
            raise ValueError("message must have 2 or 3 parts")
        key = key if key else None
        try:
            body = json.loads(json_body)
        except Exception as ex:
            logging.error('json.loads(): %s' % ex)
            body = {}
        identity = identity if identity else None
        return cls(identity=identity, key=key, body=body)
    
    def dump(self):
        if self.body is None:
            size = 0
            data='NULL'
        else:
            size = len(self.body)
            data=repr(self.body)
        print("[key:{key}][size:{size}] {data}".format(
            key=self.key,
            size=size,
            data=data,
        ), file=sys.stderr)

# ---------------------------------------------------------------------
# Runs self test of class

def test_jsonmsg (verbose):
    print(" * jsonmsg: ", end='')

    # Prepare our context and sockets
    ctx = zmq.Context()
    output = ctx.socket(zmq.DEALER)
    output.bind("ipc://jsonmsg_selftest.ipc")
    input = ctx.socket(zmq.DEALER)
    input.connect("ipc://jsonmsg_selftest.ipc")

    # Test send and receive of simple message
    jsonmsg = JsonMsg()
    jsonmsg.key = b"key"
    jsonmsg.body = b"body"
    if verbose:
        jsonmsg.dump()
    jsonmsg.send(output)

    jsonmsg2 = JsonMsg.recv(input)
    if verbose:
        jsonmsg2.dump()
    assert jsonmsg2.key == b"key"

    print("OK")

if __name__ == '__main__':
    test_jsonmsg('-v' in sys.argv)
