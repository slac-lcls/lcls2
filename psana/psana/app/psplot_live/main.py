####################################################################
# A simple program that monitors zmq messages for psplot info, store
# the info, and activate new psplot process. The main part is inter-
# active after starting these two async subprocesses:
#
# 1) run_monitor
#    Creates a DbHelper that listens for new messages through zmq 
#    socket. The default publisher (send data to DbHelper) is Kafka
#    and can be switched to zmq using ZMQ option arg. For any new
#    psplot, run_monitor generates psplot subprocess accordingly.
# 2) start_kafka_consumer
#    Starts a kafka consumer subprocess that listens to kafka publisher
#    (in user's analysis code) then passes the message to DbHelper
#    via zmq socket. Note that if ZMQ option is used instead of KAFKA,
#    user's code needs to use zmq_send method, which sends data
#    directly to DbHelper.
#
# psplot process management
# The main interactive process and run_monitor subproc communicate 
# through zmq ipc socket (psplot_live_run-monitor-subproc-pid) to exchange 
# psplot info. 
#
# The interactive session allows user to use all the functions that
# doesn't start with "_" in their name.
#
# Example run on s3df
# To start monitoring two plots,
# psplot_live ANDOR ATMOPAL
# On another node, submit an analysis job that generates these two plots
# cd psana/psana/tests
# sbatch submit_run_andor.sh rixc00221 49
####################################################################

from psana.app.psplot_live.db import *
from psana.app.psplot_live.subproc import SubprocHelper
from psana.psexp.zmq_utils import ClientSocket
from kafka import KafkaConsumer
from typing import List
import json
import typer
import asyncio
import IPython
import psutil
import atexit
import os


proc = SubprocHelper()
runner = None

def _kill_pid(pid, timeout=3, verbose=False):
    def on_terminate(proc):
        if verbose:
            print("process {} terminated with exit code {}".format(proc, proc.returncode))
    procs = [psutil.Process(pid)] + psutil.Process(pid).children()
    for p in procs:
        p.terminate()
    gone, alive = psutil.wait_procs(procs, timeout=timeout, callback=on_terminate)
    for p in alive:
        p.kill()

def _exit_handler():
    print('Cleaning up subprocesess...')
    for pid in proc.pids():
        _kill_pid(pid)
atexit.register(_exit_handler)


####################################################################
# Two subprocesses created by main.
####################################################################
KAFKA_MAX_POLL_INTERVAL_MS = 500000
KAFKA_MAX_POLL_RECORDS = 50
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "psplot_live")
KAFKA_BOOTSTRAP_SERVER = os.environ.get("KAFKA_BOOTSTRAP_SERVER", "172.24.5.240:9094")
async def start_kafka_consumer(socket_name):
    print(f'Connecting to kafa...')
    consumer = KafkaConsumer(bootstrap_servers=[KAFKA_BOOTSTRAP_SERVER], 
            max_poll_interval_ms=KAFKA_MAX_POLL_INTERVAL_MS, 
            max_poll_records=KAFKA_MAX_POLL_RECORDS)
    consumer.topics()
    consumer.subscribe([KAFKA_TOPIC])
    print(f'Connected to kafka at {KAFKA_BOOTSTRAP_SERVER}')
    sub = ClientSocket(socket_name)
    for msg in consumer:
        try:
            info = json.loads(msg.value)
            message_type = msg.topic
            # add monitoring message type
            info['msgtype'] = MonitorMsgType.PSPLOT
            sub.send(info)
            obj = sub.recv()
            print(f'Received {obj} from db-zmq-server')
        except Exception as e:
            print("Exception processing Kafka message.")
            print(e)

class MonitorMsgType:
    PSPLOT=0
    RERUN=1
    QUERY=2
    DONE=3
    DELETE=4

async def run_monitor(plotnames, socket_name):
    runnum, node, port = (0, None, None)
    db = DbHelper()
    db.connect(socket_name)
    while True:
        obj = db.recv()
        
        # Received data are either from 1) users' job or from 2) show(slurm_job_id) cmd. 
        # For 2), we need to look at the previous request for this slurm_job_id
        # and call psplot using the parameters from the request. We set the force_flag
        # so the check for new run is overridden. 
        # For 1), we use the job detail (exp, runnum, etc) as sent via the request
        # Jobs sent this way will only get plotted when it's new.  
        force_flag = False
        msgtype = obj['msgtype']
        include_instance = False
        if msgtype == MonitorMsgType.RERUN:
            db_instance = db.get(obj['force_rerun_instance_id'])
            force_flag = True
            new_slurm_job_id, new_exp, new_runnum, new_node, new_port, _, _ = db_instance
            print(f'Force rerun with {new_exp}:r{new_runnum} {new_node}:{new_port} jobid:{new_slurm_job_id}')
        elif msgtype == MonitorMsgType.PSPLOT:
            instance_id = db.save(obj)
            new_exp, new_runnum, new_node, new_port, new_slurm_job_id = obj['exp'], obj['runnum'], obj['node'], obj['port'], obj['slurm_job_id']
        elif msgtype == MonitorMsgType.QUERY:
            print(f'received a query')
            include_instance = True
        elif msgtype == MonitorMsgType.DELETE:
            instance_id = obj['instance_id']
            print(f'received a remove request for {instance_id=}')
            db.delete(instance_id)

        if msgtype in (MonitorMsgType.PSPLOT, MonitorMsgType.RERUN):
            if new_node != node or new_port != port or new_runnum > runnum or force_flag:
                exp, runnum, node, port, slurm_job_id = (new_exp, new_runnum, new_node, new_port, new_slurm_job_id)
                def set_pid(pid):
                    db.set(instance_id, DbHistoryColumns.PID, pid)
                    db.set(instance_id, DbHistoryColumns.STATUS, DbHistoryStatus.PLOTTED)
                    print(f'set pid:{pid}')
                # The last argument passed to psplot is the EXTRA info attached to the process.
                # The info is used to display what this psplot process is associated with.
                # Note that we only send hostname w/o the domain for node argument
                hostname_only = node.split('.')[0]
                cmd = f"psplot -s {node} -p {port} {' '.join(plotnames)} {instance_id},{exp},{runnum},{hostname_only},{port},{slurm_job_id}"
                print(cmd)
                await proc._run(cmd, callback=set_pid)
                if not force_flag:
                    print(f'Received new {exp}:r{runnum} {node}:{port} jobid:{slurm_job_id}', flush=True)
            else:
                print(f'Received old {new_exp}:r{new_runnum} {new_node}:{new_port} jobid:{new_slurm_job_id}. To reactivate, type: show({instance_id})', flush=True)
                db.set(instance_id, DbHistoryColumns.SLURM_JOB_ID, new_slurm_job_id)
        
        reply = {'msgtype': MonitorMsgType.DONE}
        db.send(reply, include_instance=include_instance)
####################################################################


####################################################################
# Interactive session functions
####################################################################
class Runner():
    def __init__(self, socket_name):
        self.sub = ClientSocket(socket_name)

    def show(self, instance_id):
        data = {'msgtype': MonitorMsgType.RERUN, 'force_rerun_instance_id':instance_id}
        self.sub.send(data)
        obj = self.sub.recv()
        print(f'Main received {obj} from db-zmq-server')

    def query_db(self):
        data = {'msgtype': MonitorMsgType.QUERY}
        self.sub.send(data)
        reply = self.sub.recv()
        return reply['instance']

    def list_proc(self):
        # Get all psplot process form the main process' database
        data = self.query_db()
        # 1: slurm_job_id1, rixc00221, 49, sdfmilan032, 12301, pid, DbHistoryStatus.PLOTTED
        headers = ["ID", "SLURM_JOB_ID", "EXP", "RUN", "NODE", "PORT", "STATUS"]
        format_row = "{:<5} {:<12} {:<10} {:<5} {:<35} {:<5} {:<10}"
        print(format_row.format(*headers))
        for instance_id, info in data.items():
            psplot_subproc_pid = info[-2]

            # Check if no. of subprocess created by psplot is two. We'll
            # kill the process and update the database if it's not the
            # case (due to users close the plot window).
            procs = psutil.Process(psplot_subproc_pid).children()
            if len(procs) != 2:
                kill(instance_id)
                continue

            # Do not display PID (last column) in pinfo
            row = [instance_id] + info[:-2] + [DbHistoryStatus.get_name(info[-1])]
            try:
                print(format_row.format(*row))
            except Exception as e:
                print(e)
                print(f'{row=}')
        
    def kill(self, instance_id, timeout=3):
        data = self.query_db()

        if instance_id not in data:
            print(f"could not locate instance_id:{instance_id}")
            return

        pid = data[instance_id][-2]
        _kill_pid(pid, timeout=timeout)
        # Remove the killed process from db instance
        data = {'msgtype': MonitorMsgType.DELETE, 'instance_id': instance_id}
        self.sub.send(data)
        reply = self.sub.recv()

    def kill_all(self):
        data = self.query_db()
        for instance_id, pinfo in data.items():
            print(f'kill {instance_id} (pid:{pinfo[-2]})')
            kill(instance_id)

####################################################################


def main(plotnames: List[str], connection_type: str = "KAFKA", subproc: str = "main", socket_name: str = "", debug: bool = False):
    if subproc == "main":
        socket_name = DbHelper.get_socket()
        cmd = f"psplot_live {' '.join(plotnames)} --subproc monitor --socket-name {socket_name}"
        if debug:
            cmd = f"xterm -hold -e {cmd}"
        asyncio.run(proc._run(cmd))

        conn_type = getattr(DbConnectionType, connection_type)
        if conn_type == DbConnectionType.KAFKA:
            cmd = f"psplot_live {' '.join(plotnames)} --subproc kafka --socket-name {socket_name}"
            if debug:
                cmd = f"xterm -hold -e {cmd}"
            asyncio.run(proc._run(cmd))
        global runner 
        runner = Runner(socket_name)
        IPython.embed()
    elif subproc == "kafka":
        asyncio.run(start_kafka_consumer(socket_name))
    elif subproc == "monitor":
        asyncio.run(run_monitor(plotnames, socket_name))
    else:
        print(f'Error: unsupported subprocess {subproc}')

def start():
    typer.run(main)

def ls():
    if runner is None: return
    runner.list_proc()

def kill(instance_id):
    if runner is None: return
    runner.kill(instance_id)

def show(instance_id):
    if runner is None: return
    runner.show(instance_id)

def kill_all():
    if runner is None: return
    runner.kill_all()


if __name__ == "__main__":
    start()


