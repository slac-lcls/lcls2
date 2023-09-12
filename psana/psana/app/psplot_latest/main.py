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


app = typer.Typer()


async def run_monitor(detname):
    runnum, node, port = (0, None, None)
    proc = SubprocHelper()
    db = DbHelper(port=4242)
    while True:
        obj = db.get_db_info()
        if obj['node'] != node or obj['port'] != port or obj['runnum'] > runnum:
            runnum, node, port = (obj['runnum'], obj['node'], obj['port'])
            print(f'Received new {runnum=} {node=} {port=}', flush=True)
            cmd = f"psplot -s {node} -p {port} {detname}"
            await proc._run(cmd)
        else:
            print(f'Received old {obj}')


@app.command()
def monitor(detname: str):
    asyncio.run(run_monitor(detname))
    

@app.command()
def start(detname: str):
    cmd = f"xterm -hold -e python main.py monitor {detname}"
    proc = SubprocHelper()
    asyncio.run(proc._run(cmd, prv='sys'))
    IPython.embed()


def show():
    data = [[pid, " ".join(psutil.Process(pid).cmdline())] for pid in psutil.pids()]
    headers = ["pid", "cmd"]
    format_row = "{:<12} {:<25}"
    print(format_row.format(*headers))
    for row in data:
        if row[1].find("psplot") > -1:
            print(format_row.format(*row))

if __name__ == "__main__":
    app()


