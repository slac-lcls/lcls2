####################################################################
# psbatch 
# Usage: psbatch
# Parameters:
#   - subcommand:   actions to be performed on the inputs
#   - inp_file:     input file with resources 
####################################################################

from typing import List
import json
import typer
import time
from psdaq.slurm.utils import SbatchManager, SbatchParms
from psdaq.slurm.db import DbHelper, DbHistoryStatus, DbHistoryColumns
import asyncio
import atexit
import psutil
import IPython


sbman = SbatchManager()
db = DbHelper()
runner = None


class Runner():
    def __init__(self, ):
        pass
    
    def list_jobs(self):
        # Get all psplot process form the main process' database
        data = db.instance
        # 1: sbparms, cmd, DbHistoryStatus.PLOTTED
        headers = ["ID", "SLURM_JOB_ID", "NODE", "CMD", "STATUS"]
        format_row = "{:<5} {:<12} {:<14} {:<35} {:<10}"
        print(format_row.format(*headers))
        for instance_id, info in data.items():
            slurm_job_id, sbparms, cmd, status = info

            row = [instance_id, slurm_job_id, ','.join(sbparms.nodelist), cmd] + [DbHistoryStatus.get_name(status)]
            try:
                print(format_row.format(*row))
            except Exception as e:
                print(e)
                print(f'{row=}')

    def submit(self, sbparms, cmd):
        sbman.create_job_file(sbparms, cmd)
        sbatch_cmd = f"sbatch {sbman.job_file}"
        def save_to_db(slurm_job_id):
            data = {'sbparms': sbparms,
                    'cmd': cmd,
                    'slurm_job_id': slurm_job_id
                    }
            db.save(data)
        asyncio.run(sbman.run(sbatch_cmd, callback=save_to_db))

def main(subcommand: str,
        inp_file: str,
        debug: bool = False
        ):
    global runner
    runner = Runner()
    with open('slurm.json', 'r') as f:
        data =json.loads(f.read())
    for item in data['job_details']:
        sbparms = SbatchParms(partition="anaq", nodelist=[item["node"]], ntasks=1) 
        for cmd in item['cmds']:
            cmd = "srun -N1 -n1 "+cmd
            print(f'submit {cmd}')
            runner.submit(sbparms, cmd)
    IPython.embed()

def ls():
    if runner is None: return
    runner.list_jobs()

def cancel(instance_id):
    if runner is None: return
    slurm_job_id = db.get(instance_id)[DbHistoryColumns.SLURM_JOB_ID]
    sbatch_cmd = f"scancel {slurm_job_id}"
    db.set(instance_id, DbHistoryColumns.STATUS, DbHistoryStatus.CANCELLED)
    asyncio.run(sbman.run(sbatch_cmd))

def restart(instance_id):
    if runner is None: return
    _, sbparms, cmd, _ = db.get(instance_id)
    db.set(instance_id, DbHistoryColumns.STATUS, DbHistoryStatus.REPLACED)
    runner.submit(sbparms, cmd)

def stop():
    if runner is None: return
    data = db.instance
    for instance_id, info in data.items():
        if info[DbHistoryColumns.STATUS] == DbHistoryStatus.SUBMITTED:
            cancel(instance_id)

def start():
    if runner is None: return
    data = db.instance
    for instance_id, info in data.items():
        if info[DbHistoryColumns.STATUS] == DbHistoryStatus.CANCELLED:
            restart(instance_id)



    
def _do_main():
    typer.run(main)

if __name__ == "__main__":
    start()