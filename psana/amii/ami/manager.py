import sys
import zmq
import argparse
from ami.base import ZmqPorts, ZmqConfig, ZmqCollector


class Manager(ZmqCollector):
    def __init__(self, name, host_config, zmqctx=None):
        super(__class__, self).__init__("manager", host_config, zmqctx)
        self.set_datagram_handler(self.publish)
        #self.command = self._zmqctx.socket(zmq.REP)
        #self.command.bind(self.host_config.command_address())
        #self.graph = self._zmqctx.socket(zmq.PUB)
        #self.graph.bind(self.host_config.graph_address())

    def publish(self, msg):
        print(msg.payload)


def main():
    parser = argparse.ArgumentParser(description='AMII Manager App')

    parser.add_argument(
        '-H',
        '--host',
        default='*',
        help='interface the AMII manager listens on (default: all)'
    )

    parser.add_argument(
        '-p',
        '--platform',
        type=int,
        default=0,
        help='platform number of the AMII - selects port range to use (default: 0)'
    )

    args = parser.parse_args()
    manager_cfg = ZmqConfig(
        platform=args.platform,
        binds={zmq.PULL: (args.host, ZmqPorts.FinalCollector),
        zmq.SUB: (args.host, ZmqPorts.Graph), zmq.REP: (args.host, ZmqPorts.Command)}, connects={}
    )

    try:
        manager = Manager("manager", manager_cfg)
        manager.run()
    except KeyboardInterrupt:
        print("Worker killed by user...")
        return 0


if __name__ == '__main__':
    sys.exit(main())
