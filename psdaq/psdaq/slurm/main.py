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
import psutil
import copy
import socket
from psdaq.slurm.utils import SbatchManager
from psdaq.slurm.subproc import SubprocHelper
import os, sys
from psdaq.slurm.config import Config
from IPython import embed


sbman = SbatchManager()
proc = SubprocHelper()
runner = None
LOCALHOST = socket.gethostname()


class Runner:
    def __init__(self, configfilename):
        # Allowing users' code to do relative 'import' in config file
        sys.path.append(os.path.dirname(configfilename))
        config_dict = {"platform": None, "config": None}
        try:
            exec(
                compile(open(configfilename).read(), configfilename, "exec"),
                {},
                config_dict,
            )
        except:
            print("Error parsing configuration file:", sys.exc_info()[1])
        self.platform = config_dict["platform"]
        # Check if we are getting main or derived config file
        if config_dict["config"] is None:
            config = Config(config_dict["main_config"])
            self.config = config.main_config
        else:
            self.config = config_dict["config"].select_config
        # Find xpm number
        self.xpm_id = 99
        if "control" in self.config:
            cmd_tokens = self.config["control"]["cmd"].split()
            for cmd_index, cmd_token in enumerate(cmd_tokens):
                if cmd_token == "-x":
                    self.xpm_id = int(cmd_tokens[cmd_index + 1])

        sbman.set_attr("platform", self.platform)

    def parse_config(self):
        """Extract commands from the cnf file"""
        use_feature = True
        for config_id, config_detail in self.config.items():
            if "host" in config_detail:
                use_feature = False

        if use_feature:
            node_features = sbman.get_node_features()
        else:
            node_features = None

        data = {}
        for config_id, config_detail in self.config.items():
            config_detail["comment"] = sbman.get_comment(
                self.xpm_id, self.platform, config_id
            )
            if use_feature:
                found_node = None
                for node, features in node_features.items():
                    for feature, occupied in features.items():
                        if occupied:
                            continue
                        if config_id == feature:
                            node_features[node][feature] = 1
                            found_node = node
                            break
                if not found_node:
                    node = "localhost"
                else:
                    node = found_node
            else:
                if "host" in config_detail:
                    node = config_detail["host"]
                else:
                    node = LOCALHOST
            if node not in data:
                job_details = {}
                job_details[config_id] = config_detail
                data[node] = job_details
            else:
                job_details = data[node]
                if config_id in job_details:
                    msg = f"Error: cannot create more than one {config_id} on {node}"
                    raise NameError(msg)
                else:
                    job_details[config_id] = config_detail
        self.sbjob = data
        self.node_features = node_features
        return

    def list_jobs(self):
        cmd = ""
        user = os.environ.get("USER", "")
        if not user:
            print(f"Cannot list jobs for user. $USER variable is not set.")
        else:
            if sbman.as_step:
                cmd = "sacct --format=JobIDRaw,JobName%15,User,State,Start,Elapsed%9,NNodes%5,NodeList,Comment"
            else:
                cmd = f'squeue -u {user} -o "%10i %15j %8u %8T %20S %10M %6D %R %k"'
        cmd = f"xterm -fa 'Source Code Pro' -geometry 125x31+15+15 -e watch -n 5 --no-title '{cmd}'"
        asyncio.run(proc.run(cmd))

    def submit(self):
        cmd = "sbatch << EOF\n" + sbman.sb_script + "\nEOF\n"
        asyncio.run(proc.run(cmd, wait_output=True))


def main(
    subcommand: str,
    cnf_file: str,
    as_step: bool = False,
    interactive: bool = False,
    verbose: bool = False,
    unique_ids: str = None,
):
    global runner
    runner = Runner(cnf_file)
    runner.parse_config()
    sbman.as_step = as_step
    sbman.verbose = verbose
    if subcommand == "start":
        start(unique_ids=unique_ids)
    elif subcommand == "stop":
        stop(unique_ids=unique_ids)
    elif subcommand == "restart":
        restart(unique_ids=unique_ids)
    elif subcommand == "status":
        if interactive:
            ls()
        else:
            show_status()
    else:
        print(f"Unrecognized subcommand: {subcommand}")
    if interactive:
        embed()


def _select_config_ids(unique_ids):
    config_ids = list(runner.config.keys())
    if unique_ids is not None:
        config_ids = unique_ids.split(",")
    return config_ids


def exists(unique_ids=None):
    """Check if the config matches any existing jobs"""
    job_exists = False
    job_details = sbman.get_job_info(noheader=True)

    config_ids = _select_config_ids(unique_ids)

    for config_id in config_ids:
        comment = sbman.get_comment(runner.xpm_id, runner.platform, config_id)
        if comment in job_details:
            job_exists = True
            break
    return job_exists


def _check_unique_ids(unique_ids):
    """Check user's input unique IDs with cnf file"""
    if unique_ids is None:
        return True
    config_ids = unique_ids.split(",")
    for config_id in config_ids:
        if config_id not in runner.config:
            msg = f"Error: cannot locate {config_id} in the cnf file"
            raise RuntimeError(msg)
    return True


def start(unique_ids=None, skip_check_exist=False):
    _check_unique_ids(unique_ids)
    if exists(unique_ids=unique_ids) and not skip_check_exist:
        msg = "Error: found one or more running jobs using the same resources"
        raise RuntimeError(msg)
    if sbman.as_step:
        sbman.generate_as_step(runner.sbjob, runner.node_features)
        runner.submit()
    else:
        config_ids = _select_config_ids(unique_ids)
        for node, job_details in runner.sbjob.items():
            for job_name, details in job_details.items():
                if job_name in config_ids:
                    sbman.generate(node, job_name, details, runner.node_features)
                    runner.submit()


def ls():
    if runner is None:
        return
    runner.list_jobs()


def show_status():
    job_details = sbman.get_job_info(noheader=True)
    print("%20s %12s %10s %40s" % ("Host", "UniqueID", "Status", "Command+Args"))
    for config_id, detail in runner.config.items():
        comment = sbman.get_comment(runner.xpm_id, runner.platform, config_id)
        if comment in job_details:
            job_detail = job_details[comment]
            print(
                "%20s %12s %10s %40s"
                % (
                    job_detail["nodelist"],
                    job_detail["job_name"],
                    job_detail["state"],
                    detail["cmd"],
                )
            )


def cancel(slurm_job_id):
    if runner is None:
        return
    output = sbman.call_subprocess("scancel", str(slurm_job_id))


def stop(unique_ids=None):
    """Stops running job using their comment.

    Each job is submitted with their unique comment. We can stop all the processes
    by looking at the given cnf and match the comment (see below for detail) with
    comment returned by slurm."""
    if runner is None:
        return
    _check_unique_ids(unique_ids)
    job_details = sbman.get_job_info(noheader=True)

    if unique_ids is not None:
        config_ids = unique_ids.split(",")
    else:
        config_ids = list(runner.config.keys())

    for config_id in config_ids:
        comment = sbman.get_comment(runner.xpm_id, runner.platform, config_id)
        if comment in job_details:
            cancel(job_details[comment]["job_id"])
        else:
            print(
                f"Warning: cannot stop {config_id} ({comment}). There is no job with this ID found."
            )


def restart(unique_ids=None):
    stop(unique_ids=unique_ids)
    start(unique_ids=unique_ids, skip_check_exist=True)


def _do_main():
    typer.run(main)


if __name__ == "__main__":
    start()
