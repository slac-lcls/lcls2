from typing import List
import json
import typer
from typing_extensions import Annotated
import time
import asyncio
import psutil
import copy
import socket
from psdaq.slurm.utils import SbatchManager
from psdaq.slurm.subproc import SubprocHelper
import os, sys, errno
from subprocess import Popen
from psdaq.slurm.config import Config

LOCALHOST = socket.gethostname()
DAQBATCH_SCRIPT = "submit_daqbatch.sh"
MAX_RETRIES = 30


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise


class Runner:
    # paths
    PATH_XTERM = "/usr/bin/xterm"
    PATH_TELNET = "/usr/bin/telnet"
    PATH_LESS = "/usr/bin/less"
    PATH_CAT = "/bin/cat"

    def __init__(self, configfilename, as_step=False, verbose=False, output=None):
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
        self.configfilename = configfilename
        # Check if we are getting main or derived config file
        if config_dict["config"] is None:
            config = Config(config_dict["procmgr_config"])
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

        self.sbman = SbatchManager(configfilename, self.platform, as_step, verbose, output=output)
        self.proc = SubprocHelper()
        self.parse_config()

    def parse_config(self):
        """Extract commands from the cnf file
        Data are stored in:
        1. sbjob: dictionary containing cmd and its details per node basis
        2. node_features: list of features per node as obtained by sinfo
        """
        use_feature = True
        for config_id, config_detail in self.config.items():
            # TODO: This needs fixing. We force 'host' keyword at reading in.
            if "host" in config_detail:
                use_feature = False

        if use_feature:
            node_features = self.sbman.get_node_features()
        else:
            node_features = None

        data = {}
        for config_id, config_detail in self.config.items():
            config_detail["comment"] = self.sbman.get_comment(
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
                    node = LOCALHOST
                else:
                    node = found_node
            else:
                node = config_detail["host"]
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

    def submit(self):
        with open(DAQBATCH_SCRIPT, "w") as f:
            f.write(self.sbman.sb_script)
        cmd = f"sbatch {DAQBATCH_SCRIPT}"
        asyncio.run(self.proc.run(cmd, wait_output=True))

    def _select_config_ids(self, unique_ids):
        config_ids = list(self.config.keys())
        if unique_ids is not None:
            config_ids = unique_ids.split(",")
        return config_ids

    def _exists(self, unique_ids=None):
        """Check if the config matches any existing jobs"""
        job_exists = False
        job_details = self.sbman.get_job_info()

        config_ids = self._select_config_ids(unique_ids)

        for config_id in config_ids:
            comment = self.sbman.get_comment(self.xpm_id, self.platform, config_id)
            if comment in job_details:
                job_exists = True
                break
        return job_exists

    def _check_unique_ids(self, unique_ids):
        """Check user's input unique IDs with cnf file"""
        if unique_ids is None:
            return True
        config_ids = unique_ids.split(",")
        for config_id in config_ids:
            if config_id not in self.config:
                msg = f"Error: cannot locate {config_id} in the cnf file"
                raise RuntimeError(msg)
        return True

    def show_status(self, quiet=False):
        job_details = self.sbman.get_job_info()
        result_list = []
        if not quiet:
            print(
                "%20s %12s %10s %40s" % ("Host", "UniqueID", "Status", "Command+Args")
            )
        for config_id, detail in self.config.items():
            comment = self.sbman.get_comment(self.xpm_id, self.platform, config_id)
            statusdict = {}
            statusdict["showId"] = config_id
            if comment in job_details:
                job_detail = job_details[comment]
                statusdict["status"] = job_detail["state"]
                statusdict["host"] = job_detail["nodelist"]
                statusdict["logfile"] = job_detail["logfile"]
                statusdict["job_id"] = job_detail["job_id"]
                if not quiet:
                    print(
                        "%20s %12s %10s %40s"
                        % (
                            job_detail["nodelist"],
                            job_detail["job_name"],
                            job_detail["state"],
                            detail["cmd"],
                        )
                    )
            else:
                statusdict["status"] = "COMPLETED"
                nodelist = LOCALHOST
                if "host" in detail:
                    nodelist = detail["host"]
                statusdict["host"] = nodelist
                statusdict["logfile"] = ""
                statusdict["job_id"] = ""
                if not quiet:
                    print(
                        "%20s %12s %10s %40s"
                        % (
                            nodelist,
                            config_id,
                            "COMPLETED",
                            detail["cmd"],
                        )
                    )
            # add dictionary to list
            result_list.append(statusdict)
        return result_list

    def _cancel(self, slurm_job_id):
        output = self.sbman.call_subprocess("scancel", str(slurm_job_id))

    def start(self, unique_ids=None, skip_check_exist=False):
        self._check_unique_ids(unique_ids)
        if self._exists(unique_ids=unique_ids) and not skip_check_exist:
            msg = "Error: found one or more running jobs using the same resources"
            raise RuntimeError(msg)
        if self.sbman.as_step:
            self.sbman.generate_as_step(self.sbjob, self.node_features)
            self.submit()
        else:
            config_ids = self._select_config_ids(unique_ids)
            for node, job_details in self.sbjob.items():
                for job_name, details in job_details.items():
                    if job_name in config_ids:
                        self.sbman.generate(node, job_name, details, self.node_features)
                        self.submit()

    def stop(self, unique_ids=None, skip_wait=False, verbose=False):
        """Stops running job using their comment.

        Each job is submitted with their unique comment. We can stop all the processes
        by looking at the given cnf and match the comment (see below for detail) with
        comment returned by slurm."""
        self._check_unique_ids(unique_ids)
        job_details = self.sbman.get_job_info()

        if unique_ids is not None:
            config_ids = unique_ids.split(",")
        else:
            config_ids = list(self.config.keys())

        job_states = {}
        for config_id in config_ids:
            comment = self.sbman.get_comment(self.xpm_id, self.platform, config_id)
            if comment in job_details:
                self._cancel(job_details[comment]["job_id"])
                job_states[job_details[comment]["job_id"]] = None
            else:
                if verbose:
                    print(
                        f"Warning: cannot stop {config_id} ({comment}). There is no job with this ID found."
                    )

        # Wait until all cancelled jobs reach CANCELLED state
        if not skip_wait:
            for i in range(MAX_RETRIES):
                for job_id, _ in job_states.items():
                    results = self.sbman.get_job_info_byid(job_id, ["JobState"])
                    if "JobState" in results:
                        job_states[job_id] = results["JobState"]
                active_jobs = [job_id for job_id, job_state in job_states.items() if job_state != 'CANCELLED']
                if len(active_jobs) == 0:
                    break
                if i == 0: print(f'Waiting for slurm jobs to complete...')
                time.sleep(3)


    def restart(self, unique_ids=None, verbose=False):
        self.stop(unique_ids=unique_ids, skip_wait=True, verbose=verbose)
        self.start(unique_ids=unique_ids, skip_check_exist=True)

    def spawnConsole(self, config_id, ldProcStatus, large=False):
        rv = 1  # return value (0=OK, 1=ERR)
        job_id = ""
        for statusdict in ldProcStatus:
            if statusdict["showId"] == config_id:
                job_id = statusdict["job_id"]

        if not job_id:
            print("spawnConsole: process '%s' not found" % config_id)
        else:
            try:
                cmd = f"sattach {job_id}.0"
                if large:
                    args = [
                        self.PATH_XTERM,
                        "-bg",
                        "midnightblue",
                        "-fg",
                        "white",
                        "-fa",
                        "18",
                        "-T",
                        config_id,
                        "-e",
                        cmd,
                    ]
                else:
                    args = [self.PATH_XTERM, "-T", config_id, "-e", cmd]
                Popen(args)
            except:
                print("spawnConsole failed for process '%s'" % config_id)
            else:
                rv = 0
        return rv

    def spawnLogfile(self, config_id, ldProcStatus, large=False):
        rv = 1  # return value (0=OK, 1=ERR)
        logfile = ""
        for statusdict in ldProcStatus:
            if statusdict["showId"] == config_id:
                logfile = statusdict["logfile"]

        if not os.path.exists(logfile) or not logfile:
            print(f"spawnLogfile: process {config_id} logfile not found ({logfile})")
        else:
            try:
                if large:
                    args = [
                        self.PATH_XTERM,
                        "-bg",
                        "midnightblue",
                        "-fg",
                        "white",
                        "-fa",
                        "18",
                        "-T",
                        config_id,
                        "-e",
                        self.PATH_LESS,
                        "+F",
                        logfile,
                    ]
                else:
                    args = [
                        self.PATH_XTERM,
                        "-T",
                        config_id,
                        "-e",
                        self.PATH_LESS,
                        "+F",
                        logfile,
                    ]
                Popen(args)
            except:
                print("spawnLogfile failed for process '%s'" % uniqueid)
            else:
                rv = 0
        return rv


def main(
    subcommand: Annotated[
        str, typer.Argument(help="Available options: [start, stop, restart, status]")
    ],
    cnf_file: Annotated[
        str, typer.Argument(help="Configuration file with .py extension.")
    ],
    unique_ids: Annotated[
        str,
        typer.Argument(
            help="A comma separated string containing selected processes (e.g. timing0,teb0)."
        ),
    ] = None,
    as_step: Annotated[
        bool, typer.Option(help="Submit DAQ processes as slurm job steps.")
    ] = False,
    interactive: Annotated[
        bool,
        typer.Option(
            help="Display results in a separate window for supported subcommands."
        ),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Print out sbatch script(s) submitted by daqbatch and warnings")
    ] = False,
    output: Annotated[
        str,
        typer.Option(
            "--output", "-o",
            help="Specify output path for process log files."
        ),
    ] = None,
):
    runner = Runner(cnf_file, as_step=as_step, verbose=verbose, output=output)
    if subcommand == "start":
        runner.start(unique_ids=unique_ids)
    elif subcommand == "stop":
        runner.stop(unique_ids=unique_ids, verbose=verbose)
    elif subcommand == "restart":
        runner.restart(unique_ids=unique_ids, verbose=verbose)
    elif subcommand == "status":
        runner.show_status()
    else:
        print(f"Unrecognized subcommand: {subcommand}")
    silentremove(DAQBATCH_SCRIPT)


def _do_main():
    typer.run(main)


if __name__ == "__main__":
    _do_main()
