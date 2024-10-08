import asyncio
from typing import List

import typer
from psana.app.psplot_live.db import DbHelper, DbHistoryColumns, DbHistoryStatus
from psana.app.psplot_live.subproc import SubprocHelper
from psana.app.psplot_live.utils import MonitorMsgType

app = typer.Typer()
proc = SubprocHelper()


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
        msgtype = obj["msgtype"]
        include_instance = False
        if msgtype == MonitorMsgType.RERUN:
            db_instance = db.get(obj["force_rerun_instance_id"])
            force_flag = True
            new_slurm_job_id, new_exp, new_runnum, new_node, new_port, _, _ = (
                db_instance
            )
            print(
                f"Force rerun with {new_exp}:r{new_runnum} {new_node}:{new_port} jobid:{new_slurm_job_id}"
            )
        elif msgtype == MonitorMsgType.PSPLOT:
            instance_id = db.save(obj)
            new_exp, new_runnum, new_node, new_port, new_slurm_job_id = (
                obj["exp"],
                obj["runnum"],
                obj["node"],
                obj["port"],
                obj["slurm_job_id"],
            )
        elif msgtype == MonitorMsgType.QUERY:
            print("received a query")
            include_instance = True
        elif msgtype == MonitorMsgType.DELETE:
            instance_id = obj["instance_id"]
            print(f"received a remove request for {instance_id=}")
            db.delete(instance_id)

        if msgtype in (MonitorMsgType.PSPLOT, MonitorMsgType.RERUN):
            if (
                new_node != node
                or new_port != port
                or new_runnum > runnum
                or force_flag
            ):
                exp, runnum, node, port, slurm_job_id = (
                    new_exp,
                    new_runnum,
                    new_node,
                    new_port,
                    new_slurm_job_id,
                )

                def set_pid(pid):
                    db.set(instance_id, DbHistoryColumns.PID, pid)
                    db.set(
                        instance_id, DbHistoryColumns.STATUS, DbHistoryStatus.PLOTTED
                    )
                    print(f"set pid:{pid}")

                # The last argument passed to psplot is the EXTRA info attached to the process.
                # The info is used to display what this psplot process is associated with.
                # Note that we only send hostname w/o the domain for node argument
                hostname_only = node.split(".")[0]
                cmd = f"psplot -s {node} -p {port} {' '.join(plotnames)} {instance_id},{exp},{runnum},{hostname_only},{port},{slurm_job_id}"
                print(cmd)
                await proc._run(cmd, callback=set_pid)
                if not force_flag:
                    print(
                        f"Received new {exp}:r{runnum} {node}:{port} jobid:{slurm_job_id}",
                        flush=True,
                    )
            else:
                print(
                    f"Received old {new_exp}:r{new_runnum} {new_node}:{new_port} jobid:{new_slurm_job_id}. To reactivate, type: show({instance_id})",
                    flush=True,
                )
                db.set(instance_id, DbHistoryColumns.SLURM_JOB_ID, new_slurm_job_id)

        reply = {"msgtype": MonitorMsgType.DONE}
        db.send(reply, include_instance=include_instance)


@app.callback(invoke_without_command=True)
def main(plotnames: List[str], socket_name: str):
    asyncio.run(run_monitor(plotnames, socket_name))


if __name__ == "__main__":
    app()
