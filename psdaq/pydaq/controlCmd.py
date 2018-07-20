#!/usr/bin/env python
"""
controlCmd - send a control level command via ZMQ

Author: Chris Ford <caf@slac.stanford.edu>
"""
import time
import zmq
import pprint
import argparse
from CollectMsg import CollectMsg
from ControlMsg import ControlMsg
from ControlTransition import ControlTransition as Transition

verbose = False

def getControlPorts(nodes):
    global verbose
    host = router_port = pull_port = None
    foundControl = False
    try:
        ctrl = nodes['control'][0]
    except KeyError:
        ctrl = {}
    else:
        foundControl = True
        if verbose:
            print('control node in collection:')
            pprint.pprint(ctrl)
        
    if foundControl:
        host = ctrl['connectInfo']['router_port']['adrs']
        router_port = ctrl['connectInfo']['router_port']['port']
        pull_port = ctrl['connectInfo']['pull_port']['port']
    else:
        print('No control node found in collection')

    return host, router_port, pull_port

def main():
    global verbose

    # Define commands
    command_dict = { 'configure': Transition.configure,
                     'beginrun': Transition.beginrun,
                     'enable': Transition.enable,
                     'disable': Transition.disable,
                     'endrun': Transition.endrun,
                     'unconfigure': Transition.unconfigure,
                     'getstate': ControlMsg.GETSTATE }

    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=command_dict.keys())
    parser.add_argument('-p', type=int, choices=range(0, 8), default=0, help='platform')
    parser.add_argument('-C', metavar='COLLECT_HOST', default='localhost', help='collection host')
    parser.add_argument('-v', action='store_true', help='be verbose')
    args = parser.parse_args()
    verbose = args.v

    # Prepare our context and collection mgr DEALER socket
    ctx = zmq.Context()
    collect_dealer_socket = ctx.socket(zmq.DEALER)
    collect_dealer_socket.linger = 0
    collect_dealer_socket.RCVTIMEO = 5000 # in milliseconds
    collect_dealer_socket.connect("tcp://%s:%d" % (args.C, CollectMsg.router_port(args.p)))

    # Send GETSTATE command to collection mgr
    CollectMsg(key=CollectMsg.GETSTATE).send(collect_dealer_socket)

    # Receive reply
    host = None
    try:
        collectMsg = CollectMsg.recv(collect_dealer_socket)
    except Exception as ex:
        print("CollectMsg.recv", ex)
    else:
        if verbose:
            print ("Received \"%s\"" % collectMsg.key.decode())
        # nodes
        try:
            host, router_port, pull_port = getControlPorts(collectMsg.body)
        except KeyError as ex:
            print("KeyError:", ex)
        except Exception as ex:
            print("Exception:", ex)

    # Close zmq socket
    collect_dealer_socket.close()

    if host is None:
        print("Failed to find control level ports in collection")
        return
    else:
        if verbose:
            print("Control level: host=%s  router_port=%d  pull_port=%d" %
                  (host, router_port, pull_port))

        # Prepare our control level DEALER socket
        control_dealer_socket = ctx.socket(zmq.DEALER)
        control_dealer_socket.linger = 0
        control_dealer_socket.RCVTIMEO = 5000 # in milliseconds
        control_dealer_socket.connect("tcp://%s:%d" % (host, router_port))

        # Send command
        ControlMsg(key=command_dict[args.command]).send(control_dealer_socket)

        # Receive reply
        try:
            controlMsg = ControlMsg.recv(control_dealer_socket)
        except Exception as ex:
            print("Exception:", ex)
        else:
            print ("Received \"%s\"" % controlMsg.key.decode())

        # Close zmq socket
        control_dealer_socket.close()
    
    return

if __name__ == '__main__':
    main()
