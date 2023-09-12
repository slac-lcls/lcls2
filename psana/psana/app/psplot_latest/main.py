######################################################
# Psplot uses one thread that keeps listening to db info 
# (run_no, node_name, port_no) and display psplot when 
# new info arrive using subprocess. 
######################################################
from psana.app.psplot_latest.db import DbHelper
from psana.app.psplot_latest.subproc import SubprocHelper
import typer


def main(detname: str):
    runnum, node, port = (0, None, None)
    proc = SubprocHelper()
    db = DbHelper()
    while True:
        obj = db.get_db_info()
        if obj['node'] != node or obj['port'] != port or obj['runnum'] > runnum:
            runnum, node, port = (obj['runnum'], obj['node'], obj['port'])
            print(f'Received new {runnum=} {node=} {port=}', flush=True)
            proc.start_psplot(node, port, detname)
        else:
            print(f'Received old {obj}')


if __name__ == "__main__":
    typer.run(main)


