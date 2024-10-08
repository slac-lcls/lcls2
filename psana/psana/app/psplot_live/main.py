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

import asyncio
import atexit
import os
from typing import List

import IPython
import psutil
import typer
from psana.app.psplot_live.db import DbHelper, DbHistoryStatus
from psana.app.psplot_live.subproc import SubprocHelper
from psana.app.psplot_live.utils import MonitorMsgType
from psana.psexp.zmq_utils import ClientSocket

proc = SubprocHelper()
runner = None
app = typer.Typer(add_completion=False)


def _kill_pid(pid, timeout=3, verbose=False):
    def on_terminate(proc):
        if verbose:
            print(
                "process {} terminated with exit code {}".format(
                    proc, proc.returncode
                )
            )

    procs = [psutil.Process(pid)] + psutil.Process(pid).children()
    for p in procs:
        p.terminate()
    gone, alive = psutil.wait_procs(
        procs, timeout=timeout, callback=on_terminate
    )
    for p in alive:
        p.kill()


def _exit_handler():
    print("Cleaning up subprocesess...")
    for pid in proc.pids():
        _kill_pid(pid)


atexit.register(_exit_handler)


####################################################################
# Interactive session functions
####################################################################
class Runner:
    def __init__(self, socket_name, plotnames):
        self.sub = ClientSocket(socket_name)
        self.plotnames = plotnames

    def show(self, instance_id):
        data = {
            "msgtype": MonitorMsgType.RERUN,
            "force_rerun_instance_id": instance_id,
        }
        self.sub.send(data)
        obj = self.sub.recv()
        print(f"Main received {obj} from db-zmq-server")

    def query_db(self):
        data = {"msgtype": MonitorMsgType.QUERY}
        self.sub.send(data)
        reply = self.sub.recv()
        return reply["instance"]

    def list_proc(self):
        # Get all psplot process form the main process' database
        data = self.query_db()
        # 1: slurm_job_id1, rixc00221, 49, sdfmilan032, 12301, pid, DbHistoryStatus.PLOTTED
        headers = [
            "ID",
            "SLURM_JOB_ID",
            "EXP",
            "RUN",
            "NODE",
            "PORT",
            "STATUS",
        ]
        format_row = "{:<5} {:<12} {:<10} {:<5} {:<35} {:<5} {:<10}"
        print(format_row.format(*headers))
        for instance_id, info in data.items():
            psplot_subproc_pid = info[-2]

            # Our main process creates a psplot process and for each plot GUI, another
            # subprocess. Upon checking if we only see one psplot process, we assume
            # all the plot GUIs have been closed. We'll kill the process and remove
            # it from the database.
            sprocs = psutil.Process(psplot_subproc_pid).children()

            if len(sprocs) == 1:
                kill(instance_id)
                continue

            # Do not display PID (last column) in pinfo
            row = (
                [instance_id]
                + info[:-2]
                + [DbHistoryStatus.get_name(info[-1])]
            )
            try:
                print(format_row.format(*row))
            except Exception as e:
                print(e)
                print(f"{row=}")

    def kill(self, instance_id, timeout=3):
        data = self.query_db()

        if instance_id not in data:
            print(f"could not locate instance_id:{instance_id}")
            return

        pid = data[instance_id][-2]
        _kill_pid(pid, timeout=timeout)
        # Remove the killed process from db instance
        data = {"msgtype": MonitorMsgType.DELETE, "instance_id": instance_id}
        self.sub.send(data)
        self.sub.recv()

    def kill_all(self):
        data = self.query_db()
        for instance_id, pinfo in data.items():
            print(f"kill {instance_id} (pid:{pinfo[-2]})")
            kill(instance_id)


####################################################################
@app.callback(invoke_without_command=True)
def main(
    plotnames: List[str],
    debug: bool = False,
):
    socket_name = DbHelper.get_socket()
    cmd = f"python {os.path.join(os.path.dirname(os.path.realpath(__file__)), 'psplot_monitor.py')} {' '.join(plotnames)} {socket_name}"
    if debug:
        cmd = f"xterm -hold -e {cmd}"
    asyncio.run(proc._run(cmd))

    cmd = f"python {os.path.join(os.path.dirname(os.path.realpath(__file__)), 'kafka_consumer.py')} {socket_name}"
    if debug:
        cmd = f"xterm -hold -e {cmd}"
    asyncio.run(proc._run(cmd))
    global runner
    runner = Runner(socket_name, plotnames)
    IPython.embed()


@app.command()
def ls():
    """List psplot instances with instance_id."""
    if runner is None:
        return
    runner.list_proc()


@app.command()
def kill(instance_id):
    """Close psplot window. Usage: kill(1) to close psplot with instance_id=1."""
    if runner is None:
        return
    runner.kill(instance_id)


@app.command()
def show(instance_id):
    """Re-open psplot window. Usage: show(1) to re-open psplot with instance_id=1."""
    if runner is None:
        return
    runner.show(instance_id)


@app.command()
def killall():
    """Close all psplot windows."""
    if runner is None:
        return
    runner.kill_all()


def start():
    app()


if __name__ == "__main__":
    start()
