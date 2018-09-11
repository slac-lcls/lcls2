import argparse
import zmq
import time


port   = 55560                  # Default value
zmqSrv = 'tcp://psdev7b'

parser = argparse.ArgumentParser(description='Monitor data printer')

parser.add_argument('port', type=int, nargs='?', help='Port number [%d]' % port)
parser.add_argument('-Z',   '--zmqSrv',          help='ZMQ server  [%s]' % zmqSrv)

args = parser.parse_args()

if args.port is not None:
    port = args.port

if args.zmqSrv is not None:
    zmqSrv = args.zmqSrv

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect('%s:%d' % (zmqSrv, port))
socket.setsockopt(zmq.SUBSCRIBE, b'')
print('Listening to: %s:%d' % (zmqSrv, port))

try:
    while True:
        #print (socket.recv_json())
        hostname, metrics = socket.recv_json()

        # shift timestamp from UTC to current timezone and convert to milliseconds
        ##metrics['time'] = [(t - time.altzone)*1000 for t in metrics['time']]

        line = (hostname.split('.')[0]) + ":"
        for k in metrics:
            if k == 'time':
                line +=  " %4s" % (k) + " %7s" % (str(metrics[k][0]))
            else:
                line += "  %11s" % (k) + " %9s" % (str(metrics[k][0]))
        print (line)
except KeyboardInterrupt:
    print()

