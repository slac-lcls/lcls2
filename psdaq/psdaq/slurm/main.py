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
import asyncio
import atexit
import psutil
import copy
import socket
from psdaq.slurm.utils import SbatchManager
from psdaq.slurm.db import DbHelper, DbHistoryStatus, DbHistoryColumns
from psdaq.procmgr.ProcMgr import ProcMgr, deduce_platform


sbman = SbatchManager()
db = DbHelper()
runner = None
LOCALHOST = socket.gethostname()


class Runner():
    def __init__(self, cnf_file):
        self.platform = deduce_platform(cnf_file)
        self.procmgr = ProcMgr(cnf_file, self.platform)

    def parse_cnf(self):
        """ Extract commands from the cnf file """
        data = {}
        for key, val in self.procmgr.d.items():
            node, job_name = key.split(":")
            _, _, cmd, _, _, flag, _, conda_env, env, _ = val
            if node not in data:
                job_details = {}
                job_details[job_name] = {"cmd": cmd, "flag": flag, "conda_env": conda_env, "env": env}  
                data[node] = job_details
            else:
                job_details = data[node] 
                if job_name in job_details:
                    msg = f"Error: cannot create more than one {job_name} on {node}"
                    raise NameError('HiThere')
                else:
                    job_details[job_name] = {"cmd": cmd, "flag": flag, "conda_env": conda_env, "env": env}
        return data

    def list_jobs(self):
        # Get all psplot process form the main process' database
        data = db.instance
        # 1: sbparms, cmd, DbHistoryStatus.PLOTTED
        headers = ["ID", "SLURM_JOB_ID", "STATUS"]
        format_row = "{:<5} {:<12} {:<10}"
        print(format_row.format(*headers))
        for instance_id, info in data.items():
            slurm_job_id, sb_script, status = info

            row = [instance_id, slurm_job_id, DbHistoryStatus.get_name(status)]
            try:
                print(format_row.format(*row))
            except Exception as e:
                print(e)
                print(f'{row=}')

    def submit(self):
        cmd = "sbatch << EOF\n"+sbman.sb_script+"\nEOF\n"
        def save_to_db(slurm_job_id):
            data = {'sb_script': sbman.sb_script,
                    'slurm_job_id': slurm_job_id
                    }
            db.save(data)
        asyncio.run(sbman.run(cmd, callback=save_to_db))

def main(subcommand: str,
        cnf_file: str,
        as_step: bool = False,
        ):
    global runner
    runner = Runner(cnf_file)
    sbjob = runner.parse_cnf()
    if as_step:
        sbman.generate_as_step(sbjob)
        runner.submit()
    else:
        for node, job_details in sbjob.items():
            for job_name, details in job_details.items():
                sbman.generate(node, job_name, details)
                runner.submit()

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
    _, sb_script, _ = db.get(instance_id)
    db.set(instance_id, DbHistoryColumns.STATUS, DbHistoryStatus.REPLACED)
    runner.submit()

def stop():
    if runner is None: return
    data = db.instance
    for instance_id, info in data.items():
        if info[DbHistoryColumns.STATUS] == DbHistoryStatus.SUBMITTED:
            cancel(instance_id)

def start():
    if runner is None: return
    # Make a copy of the current db dict because restart add new items,
    # which change the size of the dictionary.
    data = copy.deepcopy(db.instance)
    for instance_id, info in data.items():
        if info[DbHistoryColumns.STATUS] == DbHistoryStatus.CANCELLED:
            restart(instance_id)



    
def _do_main():
    typer.run(main)

if __name__ == "__main__":
    start()
