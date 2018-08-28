#!/usr/bin/env python
"""
control command
"""
import os
import argparse
import zmq
from collection import rep_port, create_msg

def main():

    # Define commands
    command_dict = {'reset':        'reset',
                    'plat':         'plat',
                    'alloc':        'alloc',
                    'connect':      'connect',
                    'configure':    'configured',
                    'beginrun':     'running',
                    'enable':       'enabled',
                    'disable':      'running',
                    'endrun':       'configured',
                    'unconfigure':  'connect',
                    'getstate':     '*'}

    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=command_dict.keys())
    parser.add_argument('-p', type=int, choices=range(0, 8), default=0, help='platform')
    parser.add_argument('-C', metavar='COLLECT_HOST', default='localhost', help='collection host')
    parser.add_argument('-v', action='store_true', help='be verbose')
    args = parser.parse_args()

    reply = {}
    try:
        context = zmq.Context(1)
        req = context.socket(zmq.REQ)
        req.linger = 0
        req.RCVTIMEO = 6000 # in milliseconds
        req.connect('tcp://localhost:%d' % rep_port(args.p))
        msg = create_msg(args.command)
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

        # print result
        expected = command_dict[args.command]
        if errorMessage is not None:
            print('Error: %s' % errorMessage)
        elif (expected == '*') or (expected == key):
            print(key)
        else:
            print('Error: received \"%s\", expected \"%s\"' %
                  (key, expected))
    return 

if __name__ == '__main__':
    main()
