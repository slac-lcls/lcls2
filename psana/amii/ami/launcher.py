import json
import argparse
import IPython
import numpy as np
import multiprocessing as mp
from ami.operation import Datagram
from ami.manager import Worker, Manager

def run_worker(cfg_host, cfg_port, node_slice):
    worker = Worker(cfg_host, cfg_port, node_slice)
    worker.run()


def run_gather(cfg_host, cfg_port):
    gather = Worker(cfg_host, cfg_port, node_slice=None, gather=True)
    gather.run()


def worker_main():
    parser = argparse.ArgumentParser(description='AMI2 Manager App')

    parser.add_argument(
        '-H',
        '--host',
        default='localhost',
        help='hostname of config server'
    )

    parser.add_argument(
        '-p',
        '--port',
        type=int,
        default=5556,
        help='hostname of config server'
    )

    parser.add_argument(
        '-n',
        '--num-workers',
        type=int,
        default=1,
        help='number of worker processes'
    )

    args = parser.parse_args()

    procs = []
    failed_worker = False

    try:
        for i in range(args.num_workers):
            proc = mp.Process(name='worker-slice%03d'%i, target=run_worker, args=(args.host, args.port, i))
            proc.daemon = True
            proc.start()
            procs.append(proc)

        #gather_proc = mp.Process(name='gather', target=run_gather, args=(args.host, args.port))
        #gather_proc.daemon = True
        #gather_proc.start()
        #procs.append(gather_proc)

        for proc in procs:
            proc.join()
            if proc.exitcode == 0:
                print('%s exited successfully'%proc.name)
            else:
                failed_worker = True
                print('%s exited with non-zero status code: %d'%(proc.name, proc.exitcode))

        # return a non-zero status code if any workerss died
        if failed_worker:
            return 1

        return 0
    except KeyboardInterrupt:
        print("Manager killed by user...")
        return 0


def _make_send(sender):
    def send():
        cspad_data = Datagram(count, config_id, np.random.normal(35, 5, (10, 10)))
        acq_data = Datagram(count, config_id, np.random.normal(35, 5, (10)))
        sender.send('cspad', cspad_data)
        sender.send('acq', acq_data)
    return send


def _make_send_cfg(sender):
    def send_cfg(config=None):
        if config is not None:
            sender.config = config
        sender.send_cfg()
    return send_cfg


def _make_load():
    def load(filename):
        with open(filename, 'r') as cnf:
            return json.load(cnf)
    return load


def manager_main():
    parser = argparse.ArgumentParser(description='AMI2 Config and Data Sender App')

    parser.add_argument(
        'config_file',
        default=None,
        help='configuration file to send'
    )

    parser.add_argument(
        '-c',
        '--config-id',
        type=int,
        default=0,
        help='starting config id'
    )

    parser.add_argument(
        '-n',
        '--num-slices',
        type=int,
        default=0,
        help='number of worker processes'
    )

    args = parser.parse_args()

    try:
        load = _make_load()
        config = load(args.config_file)
        manager = Manager(args.config_id, config)
        manager.start()
        return 0
    except KeyboardInterrupt:
        return 0

