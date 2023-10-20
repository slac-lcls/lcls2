######################################################
# Psplot uses asyncio that keeps listening to db info 
# (run_no, node_name, port_no) and displays psplot when 
# new info arrive using subprocess. 
######################################################
from psana.app.psplot_latest.db import DbHelper, DbHistoryStatus, DbHistoryColumns, DbConnectionType
from psana.app.psplot_latest.subproc import SubprocHelper
from psana.psexp.zmq_utils import zmq_send
from kafka import KafkaConsumer
from typing import List
import json
import typer
import asyncio
import IPython
import psutil
import atexit
import os


app = typer.Typer()
proc = SubprocHelper()
socket_name = None
conn_type = None


def _exit_handler():
    print('Cleaning up subprocesess...')
    for pid in proc.pids():
        _kill_pid(pid)
atexit.register(_exit_handler)


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
    for msg in consumer:
        try:
            info = json.loads(msg.value)
            message_type = msg.topic
            zmq_send(fake_dbase_server=socket_name, **info)
        except Exception as e:
            print("Exception processing Kafka message.")


async def run_monitor(plotnames, socket_name):
    runnum, node, port = (0, None, None)
    db = DbHelper()
    db.connect(socket_name)
    while True:
        obj = db.get_db_info()
        
        # Received data are either from 1) users' job or from 2) show(slurm_job_id) cmd. 
        # For 2), we need to look at the previous request for this slurm_job_id
        # and call psplot using the parameters from the request. We set the force_flag
        # so the check for new run is overridden. 
        # For 1), we use the job detail (exp, runnum, etc) as sent via the request
        # Jobs sent this way will only get plotted when it's new.  
        force_flag = False
        if 'force_rerun_instance_id' in obj:
            db_instance = db.get(obj['force_rerun_instance_id'])
            force_flag = True
            new_slurm_job_id, new_exp, new_runnum, new_node, new_port, _, _ = db_instance
            print(f'Force rerun with {new_exp}:r{new_runnum} {new_node}:{new_port} jobid:{new_slurm_job_id}')
        else:
            instance_id = db.save(obj)
            new_exp, new_runnum, new_node, new_port, new_slurm_job_id = obj['exp'], obj['runnum'], obj['node'], obj['port'], obj['slurm_job_id']
        
        if new_node != node or new_port != port or new_runnum > runnum or force_flag:
            exp, runnum, node, port, slurm_job_id = (new_exp, new_runnum, new_node, new_port, new_slurm_job_id)
            def set_pid(pid):
                db.set(instance_id, DbHistoryColumns.PID, pid)
                db.set(instance_id, DbHistoryColumns.STATUS, DbHistoryStatus.PLOTTED)
                print(f'set pid:{pid}')
            # The last argument passed to psplot is the EXTRA info attached to the process.
            # The info is used to display what this psplot process is associated with.
            cmd = f"psplot -s {node} -p {port} {' '.join(plotnames)} {instance_id},{exp},{runnum},{node},{port},{slurm_job_id}"
            print(cmd)
            await proc._run(cmd, callback=set_pid)
            if not force_flag:
                print(f'Received new {exp}:r{runnum} {node}:{port} jobid:{slurm_job_id}', flush=True)
        else:
            print(f'Received old {new_exp}:r{new_runnum} {new_node}:{new_port} jobid:{new_slurm_job_id}. To reactivate, type: show({instance_id})', flush=True)
            db.set(instance_id, DbHistoryColumns.SLURM_JOB_ID, new_slurm_job_id)


@app.command()
def kafka(socket_name: str):
    asyncio.run(start_kafka_consumer(socket_name))

@app.command()
def monitor(plotnames: List[str], socket_name: str):
    asyncio.run(run_monitor(plotnames, socket_name))
    
@app.command()
def start(plotnames: List[str], connection_type: str = "KAFKA"):
    global socket_name
    socket_name = DbHelper.get_socket()
    cmd = f"xterm -hold -e python main.py monitor {' '.join(plotnames)} {socket_name}"
    asyncio.run(proc._run(cmd))
    
    global conn_type
    conn_type = getattr(DbConnectionType, connection_type)
    if conn_type == DbConnectionType.KAFKA:
        cmd = f"xterm -hold -e python main.py kafka {socket_name}"
        asyncio.run(proc._run(cmd))
    IPython.embed()

def show(instance_id):
    zmq_send(fake_dbase_server=socket_name, 
            force_rerun_instance_id=instance_id)

def _get_proc_info(keyword):
    data = {}
    # Psplot creats one or more subprocesses - we only need to
    # list the pid of the main process.
    for proc in psutil.process_iter(['pid', 'cmdline']):
        if ' '.join(proc.info['cmdline']).find(keyword) > -1:
            pinfo = proc.info['cmdline'][-1].split(',')
            instance_id = int(pinfo[0])
            pid = int(proc.info['pid'])
            if instance_id in data:
                if pid < data[instance_id][-1]:
                    data[instance_id] = pinfo[1:] + [pid]
            else:
                data[instance_id] = pinfo[1:] + [pid]
    return data


def ls():
    data = _get_proc_info("psplot")
    headers = ["ID", "EXP", "RUN", "NODE", "PORT", "SLURM_JOB_ID", "PID"]
    format_row = "{:<5} {:<10} {:<5} {:<20} {:<8} {:<12} {:<10}"
    print(format_row.format(*headers))
    for instance_id, pinfo in data.items():
        row = [instance_id] + pinfo 
        # Skip other psplot processes that do not have these fields 
        # (they are not invoked by psplotdb server app).
        if len(row) != 7:
            continue
        try:
            print(format_row.format(*row))
        except Exception as e:
            print(e)
            print(f'{row=}')


def kill(instance_id, timeout=3):
    data = _get_proc_info("psplot")

    if instance_id not in data:
        print(f"could not locate instance_id:{instance_id}")
        return

    pid = data[instance_id][-1]
    _kill_pid(pid, timeout=timeout)


def _kill_pid(pid, timeout=3):
    def on_terminate(proc):
        print("process {} terminated with exit code {}".format(proc, proc.returncode))
    procs = [psutil.Process(pid)] + psutil.Process(pid).children()
    for p in procs:
        p.terminate()
    gone, alive = psutil.wait_procs(procs, timeout=timeout, callback=on_terminate)
    for p in alive:
        p.kill()



def kill_all():
    data = _get_proc_info("psplot")
    for instance_id, pinfo in data.items():
        print(f'kill {instance_id} (pid:{pinfo[-1]})')
        kill(instance_id)

if __name__ == "__main__":
    app()


