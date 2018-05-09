#!/usr/bin/env python
"""
statecmd - send a state command via ZMQ

Author: Chris Ford <caf@slac.stanford.edu>
"""
import time
import zmq
import argparse
from CMMsg import CMMsg
from ControlTransition import ControlTransition as Transition

def main():

    # Define commands
    command_dict = { 'configure': Transition.configure,
                     'beginrun': Transition.beginrun,
                     'enable': Transition.enable,
                     'disable': Transition.disable,
                     'endrun': Transition.endrun,
                     'unconfigure': Transition.unconfigure,
                     'getstate': CMMsg.GETSTATE }

    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=command_dict.keys())
    parser.add_argument('-p', type=int, choices=range(0, 8), default=0, help='platform (default 0)')
    parser.add_argument('-C', metavar='CTRL_HOST', default='localhost', help='control host')
    args = parser.parse_args()

    # Prepare our context and DEALER socket
    ctx = zmq.Context()
    cmd_socket = ctx.socket(zmq.DEALER)
    cmd_socket.linger = 0
    cmd_socket.RCVTIMEO = 5000 # in milliseconds
    cmd_socket.connect("tcp://%s:%d" % (args.C, CMMsg.router_port(args.p)))

    # Send command
    cmd_socket.send(command_dict[args.command])

    # Receive reply
    try:
        cmmsg = CMMsg.recv(cmd_socket)
    except Exception as ex:
        print(ex)
    else:
        print ("Received \"%s\"" % cmmsg.key.decode())
    return

if __name__ == '__main__':
    main()
