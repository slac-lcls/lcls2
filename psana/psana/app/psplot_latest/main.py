######################################################
# Psplot uses asyncio that keeps listening to db info 
# (run_no, node_name, port_no) and displays psplot when 
# new info arrive using subprocess. 
######################################################
from psana.app.psplot_latest.db import DbHelper, DbHistoryStatus, DbHistoryColumns
from psana.app.psplot_latest.subproc import SubprocHelper
from psana.psexp.zmq_utils import zmq_send
import typer
import asyncio
import IPython
import psutil
import atexit


app = typer.Typer()
proc = SubprocHelper()
socket_name = None


def _exit_handler():
    print('Cleaning up subprocesess...')
    for pid in proc.pids():
        kill(pid)
atexit.register(_exit_handler)


async def run_monitor(detname, socket_name):
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
        if 'force_rerun_slurm_job_id' in obj:
            key, instance_id = db.get(DbHistoryColumns.SLURM_JOB_ID, obj['force_rerun_slurm_job_id'])
            if key is None:
                print(f'Could not locate any request with this key: {key}, instance_id: {instance_id}')
            else:
                force_flag = True
                new_exp, new_runnum, new_node, new_port = key
                new_slurm_job_id = obj['force_rerun_slurm_job_id']
                print(f'Force rerun with {new_exp}:r{new_runnum} {new_node}:{new_port} jobid:{new_slurm_job_id}')
        else:
            key, instance_id = db.save(obj)
            new_exp, new_runnum, new_node, new_port, new_slurm_job_id = obj['exp'], obj['runnum'], obj['node'], obj['port'], obj['slurm_job_id']
        
        if new_node != node or new_port != port or new_runnum > runnum or force_flag:
            exp, runnum, node, port, slurm_job_id = (new_exp, new_runnum, new_node, new_port, new_slurm_job_id)
            def set_pid(pid):
                db.history[key, instance_id] = [DbHistoryStatus.PLOTTED, pid, slurm_job_id] 
                print(f'set pid:{pid}')
            cmd = f"psplot -s {node} -p {port} {detname} {exp},{runnum},{node},{port},{instance_id},{slurm_job_id}"
            await proc._run(cmd, callback=set_pid)
            if not force_flag:
                print(f'Received new {exp}:r{runnum} {node}:{port} jobid:{slurm_job_id}', flush=True)
        else:
            print(f'Received old {new_exp}:r{new_runnum} {new_node}:{new_port} jobid:{new_slurm_job_id}. To reactivate, type: show({new_slurm_job_id})', flush=True)
            db.history[key, instance_id][DbHistoryColumns.SLURM_JOB_ID] = new_slurm_job_id


@app.command()
def monitor(detname: str, socket_name: str):
    asyncio.run(run_monitor(detname, socket_name))
    

@app.command()
def start(detname: str):
    global socket_name
    socket_name = DbHelper.get_socket(port=4242)
    cmd = f"xterm -hold -e python main.py monitor {detname} {socket_name}"
    asyncio.run(proc._run(cmd))
    IPython.embed()

def show(slurm_job_id):
    zmq_send(fake_dbase_server=socket_name, 
            force_rerun_slurm_job_id=slurm_job_id)

def _get_proc_info(keyword):
    data = {}
    # Psplot creats one or more subprocesses - we only need to
    # list the pid of the main process.
    for proc in psutil.process_iter(['pid', 'cmdline']):
        if ' '.join(proc.info['cmdline']).find(keyword) > -1:
            key = tuple(proc.info['cmdline'][-1].split(','))
            pid = int(proc.info['pid'])
            if key in data:
                if pid < data[key]:
                    data[key] = pid
            else:
                data[key] = pid
    return data


def ls():
    data = _get_proc_info("psplot")
    headers = ["exp", "run", "node", "port", "instance", "slurm jobid", "pid"]
    format_row = "{:<10} {:<5} {:<20} {:<8} {:<10} {:<12} {:<10}"
    print(format_row.format(*headers))
    for key, val in data.items():
        row = list(key) + [val]
        print(format_row.format(*row))


def kill(pid, timeout=3):
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
    for info, pid in data.items():
        print(f'kill {info}')
        kill(pid)

if __name__ == "__main__":
    app()


