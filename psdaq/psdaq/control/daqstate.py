#!/usr/bin/env python
"""
daqstate command
"""
import os
import sys
import time
import zmq
from psdaq.control.collection import rep_port, create_msg, states, triggers
import pprint
import argparse

def main():

    # process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, choices=range(0, 8), default=0, help='platform (default 0)')
    parser.add_argument('-C', metavar='COLLECT_HOST', default='localhost', help='collection host')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--state', choices=states)
    group.add_argument('--transition', choices=triggers)
    args = parser.parse_args()
    platform = args.p

    # initialize zmq socket
    try:
        context = zmq.Context(1)
        req = context.socket(zmq.REQ)
        req.linger = 0
        req.RCVTIMEO = 10000 # in milliseconds
        req.connect('tcp://%s:%d' % (args.C, rep_port(platform)))
    except Exception as ex:
        print('Exception: %s' % ex)
        sys.exit(1)

    if args.state:
        # state request
        try:
            msg = create_msg('setstate.' + args.state)
            req.send_json(msg)
            reply = req.recv_json()
        except Exception as ex:
            print('Exception: %s' % ex)
        except KeyboardInterrupt:
            pass
        else:
            try:
                key = reply['header']['key']
            except KeyError:
                key = ''

            errorMessage = None
            try:
                errorMessage = reply['body']['error']
            except Exception as ex:
                pass

            if errorMessage is not None:
                print('Error: %s' % errorMessage)

    elif args.transition:
        # transition request
        try:
            msg = create_msg(args.transition)
            req.send_json(msg)
            reply = req.recv_json()
        except Exception as ex:
            print('Exception: %s' % ex)
        except KeyboardInterrupt:
            pass
        else:
            try:
                key = reply['header']['key']
            except KeyError:
                key = ''

            errorMessage = None
            try:
                errorMessage = reply['body']['error']
            except Exception as ex:
                pass

            if errorMessage is not None:
                print('Error: %s' % errorMessage)

    # get current state
    try:
        msg = create_msg('getstate')
        req.send_json(msg)
        reply = req.recv_json()
    except Exception as ex:
        print('Exception: %s' % ex)
    except KeyboardInterrupt:
        pass
    else:
        key = 'error'
        try:
            key = reply['header']['key']
        except KeyError:
            pass

        print (key)

if __name__ == '__main__':
    main()
