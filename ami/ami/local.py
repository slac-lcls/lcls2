import os
import re
import sys
import shutil
import signal
import tempfile
import argparse
import multiprocessing as mp

from ami.manager import run_manager
from ami.worker import run_worker, run_collector
from ami.client import run_client


def main():
    parser = argparse.ArgumentParser(description='AMII Single Node App')

    parser.add_argument(
        '-n',
        '--num-workers',
        type=int,
        default=1,
        help='number of worker processes (default: 1)'
    )

    parser.add_argument(
        '-l',
        '--load',
        help='saved AMII configuration to load'
    )

    parser.add_argument(
        'source',
        metavar='SOURCE',
        help='data source configuration (exampes: static://test.json, psana://exp=xcsdaq13:run=14)'
    )

    args = parser.parse_args()
    ipcdir = tempfile.mkdtemp()
    collector_addr = "ipc://%s/node_collector"%ipcdir
    upstream_addr = "ipc://%s/collector"%ipcdir
    graph_addr = "ipc://%s/graph"%ipcdir
    comm_addr = "ipc://%s/comm"%ipcdir

    procs = []
    failed_proc = False

    try:
        src_url_match = re.match('(?P<prot>.*)://(?P<body>.*)', args.source)
        if src_url_match:
            src_cfg = src_url_match.groups()
        else:
            print("Invalid data source config string:", args.source)
            return 1

        for i in range(args.num_workers):
            proc = mp.Process(
                name='worker%03d-n0'%i,
                target=run_worker,
                args=(i, src_cfg, collector_addr, graph_addr)
            )
            proc.daemon = True
            proc.start()
            procs.append(proc)

        collector_proc = mp.Process(
            name='nodecol-n0',
            target=run_collector,
            args=(0, args.num_workers, collector_addr, upstream_addr)
        )
        collector_proc.daemon = True
        collector_proc.start()
        procs.append(collector_proc)

        manager_proc = mp.Process(
            name='manager',
            target=run_manager,
            args=(collector_addr, graph_addr, comm_addr)
        )
        manager_proc.daemon = True
        manager_proc.start()
        procs.append(manager_proc)

        client_proc = mp.Process(
            name='client',
            target=run_client,
            args=(comm_addr, args.load)
        )
        client_proc.daemon = False
        client_proc.start()
        client_proc.join()

        for proc in procs:
            proc.terminate()
            proc.join()
            if proc.exitcode == 0 or proc.exitcode == -signal.SIGTERM:
                print('%s exited successfully'%proc.name)
            else:
                failed_proc = True
                print('%s exited with non-zero status code: %d'%(proc.name, proc.exitcode))

        # return a non-zero status code if any workerss died
        if client_proc.exitcode != 0:
            return client_proc.exitcode
        elif failed_proc:
            return 1


    except KeyboardInterrupt:
        print("Worker killed by user...")
        return 0
    finally:
        if ipcdir is not None and os.path.exists(ipcdir):
            shutil.rmtree(ipcdir)


if __name__ == '__main__':
    sys.exit(main())
