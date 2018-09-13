import argparse
import zmq

port = 55559                    # Default value

parser = argparse.ArgumentParser(description='ZMQ port forwarder')

parser.add_argument('-P', '--port',     type=int, help='Port number [%d]' % port)
parser.add_argument('-p', '--platform', type=int, choices=range(0, 8), default=0, help='Platform number')

args = parser.parse_args()

if args.port is None:
    port += 2 * args.platform
else:
    port = args.port

context = zmq.Context(1)
# Socket facing clients
frontend = context.socket(zmq.SUB)
frontend.bind('tcp://*:%d' % (port))
frontend.setsockopt(zmq.SUBSCRIBE, b'')
print('Front-end port:', port)

# Socket facing services
backend = context.socket(zmq.PUB)
backend.bind('tcp://*:%d' % (port + 1))
print('Back-end  port:', port + 1)

try:
    zmq.device(zmq.FORWARDER, frontend, backend)
except KeyboardInterrupt:
    print()
