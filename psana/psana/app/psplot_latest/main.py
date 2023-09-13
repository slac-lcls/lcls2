######################################################
# Psplot uses one thread that keeps listening to db info 
# (run_no, node_name, port_no) and display psplot when 
# new info arrive using subprocess. 
######################################################
from psana.app.psplot_latest.db import DbHelper
from psana.app.psplot_latest.subproc import SubprocHelper
import typer
import asyncio
import IPython
import psutil
import atexit


app = typer.Typer()
proc = SubprocHelper()


def _exit_handler():
    print('Cleaning up subprocesess...')
    for pid in proc.pids():
        kill(pid)
atexit.register(_exit_handler)


async def run_monitor(detname):
    runnum, node, port = (0, None, None)
    db = DbHelper(port=4242)
    while True:
        obj = db.get_db_info()
        if obj['node'] != node or obj['port'] != port or obj['runnum'] > runnum:
            exp, runnum, node, port = (obj['exp'], obj['runnum'], obj['node'], obj['port'])
            print(f'Received new {exp=} {runnum=} {node=} {port=}', flush=True)
            cmd = f"psplot -s {node} -p {port} {detname} exp:{exp},run:{runnum},detname:{detname},node:{node},port:{port}"
            await proc._run(cmd)
        else:
            print(f'Received old {obj}')


@app.command()
def monitor(detname: str):
    asyncio.run(run_monitor(detname))
    

@app.command()
def start(detname: str):
    cmd = f"xterm -hold -e python main.py monitor {detname}"
    asyncio.run(proc._run(cmd, prv='sys'))
    IPython.embed()


def show():
    headers = ["pid", "username", "detail"]
    format_row = "{:<8} {:<10} {:<25}"
    print(format_row.format(*headers))
    for proc in psutil.process_iter(['pid', 'cmdline', 'username']):
        if ' '.join(proc.info['cmdline']).find("psplot") > -1:
            row = [proc.info['pid'], proc.info['username'], proc.info['cmdline'][-1]]
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


if __name__ == "__main__":
    app()


