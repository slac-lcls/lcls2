#!/usr/bin/env python
"""
CM Phase 1 command
"""
import time
import zmq
import sys
from CMMsg import CMMsg

def main():

    # Process arguments
    if len(sys.argv) == 2:
        partName = sys.argv[1]
    else:
        print("Usage: %s <partition name>" % sys.argv[0])
        sys.exit(1)

    # Prepare our context and DEALER socket
    ctx = zmq.Context()
    cmd = ctx.socket(zmq.DEALER)
    cmd.linger = 0
    cmd.RCVTIMEO = 5000 # in milliseconds
    cmd.connect("tcp://localhost:5556")

    # Initiate phase 1
    newmsg = CMMsg(0, key=CMMsg.STARTPH1)
    newmsg['partName'] = partName
    newmsg.send(cmd)

    while True:
        try:
            cmmsg = CMMsg.recv(cmd)
        except Exception as ex:
            print(ex)
            return

        if cmmsg.key == CMMsg.PH1STARTED:
            print ("I: Received PH1STARTED")
            break          # Done
        else:
            print ("W: Received key \"%s\"" % cmmsg.key)
            continue

#   print ("Done")

if __name__ == '__main__':
    main()
