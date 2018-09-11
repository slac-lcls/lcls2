import argparse
import zmq

port = 55559                    # Default value

parser = argparse.ArgumentParser(description='ZMQ port forwarder')

parser.add_argument('port', type=int, nargs='?', help='Port number [%d]' % port)

args = parser.parse_args()

if args.port is not None:
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

