import os
import sys
import socket
from datetime import datetime
import subprocess
import time
import logging
import shlex
from subprocess import CalledProcessError, PIPE

LOCALHOST = socket.gethostname()
SLURM_PARTITION = "drpq"
DRP_N_RSV_CORES = int(os.environ.get("PS_DRP_N_RSV_CORES", "4"))
SCRIPTS_ROOTDIR = "/reg/g/pcds/dist/pds"
RETRYABLE_CMDS = {"sbatch", "sinfo", "scancel"}

logger = logging.getLogger(__name__)


class PSbatchSubCommand:
    START = 0
    STOP = 1
    RESTART = 2


def run_slurm_with_retries(*args, max_retries=3, retry_delay=5):
    """
    Calls a subprocess, with retries for specific Slurm commands,
    and graceful handling for 'scontrol show job'.

    Parameters:
    - *args: Command and arguments to pass to subprocess.
    - max_retries: Max number of retries before failing (only applies to retryable commands).
    - retry_delay: Delay between retries in seconds.

    Returns:
    - Decoded output string from the subprocess call, or None if suppressed.

    Raises:
    - CalledProcessError: If the command fails after all retries (unless handled gracefully).
    """
    cmd = args[0]
    is_retryable = cmd in RETRYABLE_CMDS
    is_scontrol_show_job = (
        cmd == "scontrol"
        and len(args) >= 3
        and args[1] == "show"
        and args[2] == "job"
    )

    attempt = 0
    while True:
        try:
            output = subprocess.check_output(args, stderr=PIPE).strip()
            return output.decode("utf-8")
        except CalledProcessError as e:
            cmd_str = " ".join(args)
            stderr_text = e.stderr.decode("utf-8").strip()

            if is_scontrol_show_job:
                logger.debug("Command '%s' failed (non-critical): %s", cmd_str, stderr_text)
                return None

            if not is_retryable or attempt >= max_retries:
                logger.error("Subprocess call '%s' failed.%s\nError: %s",
                             cmd_str,
                             f" Reached max retries ({max_retries})." if is_retryable else "",
                             stderr_text)
                raise

            attempt += 1
            logger.warning("Attempt %d/%d: Subprocess call '%s' failed.\nError: %s\nRetrying in %d sec...",
                           attempt, max_retries, cmd_str, stderr_text, retry_delay)
            time.sleep(retry_delay)


class SbatchManager:
    def __init__(
        self, configfilename, xpm_id, platform, station, as_step, verbose, output=None
    ):
        self.sb_script = ""
        now = datetime.now()
        self.output_prefix_datetime = now.strftime("%d_%H:%M:%S")
        if output is None:
            self.output_path = os.path.join(
                os.environ.get("HOME", ""), now.strftime("%Y"), now.strftime("%m")
            )
        else:
            self.output_path = os.path.join( output, now.strftime("%Y"), now.strftime("%m"))
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.configfilename = configfilename
        self.xpm_id = xpm_id
        self.platform = platform
        self.station = station
        self.as_step = as_step
        self.verbose = verbose
        self.user = os.environ["USER"]
        self.hutch = self.user[: self.user.find("opr")]
        self.scripts_dir = os.path.join(SCRIPTS_ROOTDIR, self.hutch, "scripts")

    def set_attr(self, attr, val):
        setattr(self, attr, val)

    def get_comment(self, config_id):
        comment = f"x{self.xpm_id}_p{self.platform}_s{self.station}_{config_id}"
        return comment

    def get_node_features(self):
        lines = run_slurm_with_retries("sinfo", "-N", "-h", "-o", '"%N %f"').splitlines()
        node_features = {}
        for line in lines:
            node, features = line.strip('"').split()
            node_features[node] = {feature: 0 for feature in features.split(",")}
        return node_features

    def get_job_info_byid(self, job_id, jobparms):
        """Returns a dictionary containing values obtained from scontrol
        with the given jobparms list. Returns {} if this job does not exist.
        """
        output = run_slurm_with_retries("scontrol", "show", "job", job_id)
        results = {}
        if output is not None:
            scontrol_lines = output.splitlines()
            for jobparm in jobparms:
                for scontrol_line in scontrol_lines:
                    scontrol_cols = scontrol_line.split()
                    for scontrol_col in scontrol_cols:
                        if scontrol_col.find(jobparm) > -1:
                            _, jobparm_val = scontrol_col.split("=")
                            results[jobparm] = jobparm_val
        return results

    def get_job_info(self, use_sacct=False):
        """
        Retrieves job information from Slurm using `squeue` or `sacct`.
        Handles transient failures with retries, logs malformed lines,
        and always returns a dictionary to ensure GUI stability.

        Returns:
            dict: Mapping of job comment strings to job detail dictionaries.
        """
        user = self.user
        if not user:
            logger.warning("Cannot list jobs: $USER is not set.")
            return {}

        try:
            if use_sacct:
                format_string = "JobID,Comment%30,JobName,State,NodeList"
                output = run_slurm_with_retries(
                    "sacct", "-u", user, "-n", f"--format={format_string}"
                )
            else:
                format_string = '"%i %k %j %T %R"'
                output = run_slurm_with_retries(
                    "squeue", "-u", user, "-h", "-o", format_string
                )
        except Exception as e:
            logger.warning("Failed to retrieve job info using Slurm: %s", str(e))
            return {}

        job_details = {}
        lines = output.splitlines()

        for job_info in lines:
            cols = job_info.strip('"').split()
            if len(cols) < 1:
                logger.debug("Skipping empty or malformed job line: %s", job_info)
                continue
            if not cols[0].isdigit():
                logger.debug("Skipping non-numeric JobID in job line: %s", job_info)
                continue

            if len(cols) == 5:
                job_id, comment, job_name, state, nodelist = cols
            elif len(cols) > 5:
                job_id, comment, job_name, state = cols[:4]
                nodelist = " ".join(cols[5:])
            else:
                logger.debug("Unexpected number of fields in job line: %s", job_info)
                continue

            # Attempt to get logfile path from scontrol
            logfile = ""
            scontrol_result = run_slurm_with_retries("scontrol", "show", "job", job_id)
            if scontrol_result is not None:
                for line in scontrol_result.splitlines():
                    if "StdOut=" in line:
                        try:
                            logfile = line.split("StdOut=")[1].strip()
                            if not logfile:
                                logfile = "unknown.log"
                        except IndexError:
                            logger.debug("Malformed StdOut line in scontrol output: %s", line)
                            logfile = "unknown.log"

            # Store job info (prefer latest job_id if duplicate comment exists)
            if comment not in job_details or int(job_id) > int(job_details[comment]["job_id"]):
                job_details[comment] = {
                    "job_id": job_id,
                    "job_name": job_name,
                    "state": state,
                    "nodelist": nodelist,
                    "logfile": logfile,
                }

        return job_details

    def get_output_filepath(self, node, job_name):
        output_filepath = os.path.join(
            self.output_path,
            self.output_prefix_datetime + "_" + node + ":" + job_name + ".log",
        )
        return output_filepath

    def get_daq_cmd(self, details, job_name):
        cmd = details["cmd"]
        if "flags" in details:
            if details["flags"].find("p") > -1:
                cmd += f" -p {repr(self.platform)}"
            if details["flags"].find("u") > -1:
                cmd += f" -u {job_name}"
        if job_name == "daqstat":
            cmd += f" {self.configfilename}"
        if self.is_drp(details["cmd"]):
            n_workers = self.get_n_cores(details) - DRP_N_RSV_CORES
            if n_workers < 1:
                n_workers = 1
            cmd += f" -W {n_workers}"
        return cmd

    def is_drp(self, cmd):
        return cmd.strip().startswith("drp ")

    def get_n_cores(self, details):
        n_cores = 1
        if "cores" in details:
            n_cores = int(details["cores"])
        return n_cores

    def get_rtprio(self, details):
        """Return '', raise, or return valid 'rtprio value'"""
        rtprio = ""
        if "rtprio" in details:
            if not details["rtprio"].isdigit():
                raise ValueError("malformed rtprio value: %s" % details["rtprio"])
            else:
                rtprio_as_int = int(details["rtprio"])
                if (rtprio_as_int < 1) or (rtprio_as_int > 99):
                    raise ValueError("rtprio not in range 1-99: %s" % details["rtprio"])
                else:
                    rtprio = details["rtprio"]
        return rtprio

    def get_jobstep_cmd(
        self, node, job_name, details, het_group=-1, with_output=False, as_step=False
    ):
        output_opt = ""
        if with_output:
            output = self.get_output_filepath(node, job_name)
            output_opt = f"--output={output} --open-mode=append "

        env_opt = "--export="

        # Inherit follows from user's account
        env_opt += "HOME"
        env_opt += ",USER"
        env_opt += ",TESTRELDIR"
        env_opt += ",CONDA_PREFIX"
        env_opt += ",CONDA_DEFAULT_ENV"
        env_opt += ",CONDA_EXE"
        env_opt += ",CONFIGDB_AUTH"
        env_opt += ",SUBMODULEDIR"

        # Build PATH and PYTHONPATH from scratch
        daq_path = "$TESTRELDIR/bin"
        daq_path += ":$CONDA_PREFIX/bin"
        daq_path += ":$CONDA_PREFIX/epics/bin/linux-x86_64"
        daq_path += ":/usr/sbin"
        daq_path += ":/usr/bin"
        daq_path += ":/sbin"
        daq_path += ":/bin"
        env_opt += ",PATH=" + daq_path
        env_opt += f",PYTHONPATH=$TESTRELDIR/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"

        # For x11 forwarding
        env_opt += ",DISPLAY"
        env_opt += ",XAUTHORITY"

        # Include any exists in setup_env.sh backdoor
        env_opt += ",$DAQMGR_EXPORT"

        # Include any exists in the configuration file
        cnf_env = ""
        found_ld_library_path = False
        if "env" in details:
            if details["env"] != "":
                envs = shlex.split(details["env"], posix=True)
                for i, env in enumerate(envs):
                    env_name, env_var = env.split("=")
                    if env_name == "LD_LIBRARY_PATH":
                        found_ld_library_path = True
                    sanitized_env = env
                    if " " in env_var:
                        sanitized_env = f'{env_name}="{env_var}"'
                    cnf_env += "," + sanitized_env
        env_opt += cnf_env

        if not found_ld_library_path:
            env_opt += ",LD_LIBRARY_PATH=$TESTRELDIR/lib"

        env_opt += " "

        het_group_opt = ""
        if het_group > -1:
            het_group_opt = f"--het-group={het_group} "

        # Generate as set of commands for srun
        daq_cmd = self.get_daq_cmd(details, job_name)

        daqlog_header = (
            f'daqlog_header {job_name} {self.platform} {node} "{daq_cmd.strip()}";'
        )

        rtprio = self.get_rtprio(details)
        rtattr = f"/usr/bin/chrt -f {rtprio} " if rtprio else ""

        cmd = f"{daqlog_header}{rtattr}{daq_cmd}"
        if "conda_env" in details:
            if details["conda_env"] != "":
                CONDA_EXE = os.environ.get("CONDA_EXE", "")
                loc_inst = CONDA_EXE.find("/inst")
                conda_profile = os.path.join(
                    CONDA_EXE[:loc_inst], "inst", "etc", "profile.d", "conda.sh"
                )
                cmd = f"source {conda_profile}; conda activate {details['conda_env']}; {cmd}"

        n_cores = self.get_n_cores(details)
        if not as_step:
            jobstep_cmd = f"srun -n1 -c{n_cores} --unbuffered --job-name={job_name} {het_group_opt}{output_opt}{env_opt}bash -c '{cmd}'"
        else:
            jobstep_cmd = (
                f"srun -n1 --exclusive --cpus-per-task={n_cores} --unbuffered --job-name={job_name} {het_group_opt}{output_opt}{env_opt}bash -c '{cmd}'"
                + "&\n"
            )
        return jobstep_cmd

    def generate_as_step(self, sbjob, node_features):
        sb_script = "#!/bin/bash\n"
        sb_script += f"#SBATCH --partition={SLURM_PARTITION}" + "\n"
        sb_script += "#SBATCH --job-name=main" + "\n"
        output = self.get_output_filepath(LOCALHOST, "slurm")
        sb_script += f"#SBATCH --output={output}" + "\n"
        sb_header = ""
        sb_steps = ""
        for het_group, (node, job_details) in enumerate(sbjob.items()):
            if node_features is None:
                sb_header += (
                    f"#SBATCH --nodelist={node} --ntasks={len(job_details.keys())}"
                    + "\n"
                )
            else:
                features = ",".join(node_features[node].keys())
                sb_header += (
                    f"#SBATCH --constraint={features} --ntasks={len(job_details.keys())}"
                    + "\n"
                )

            if het_group < len(sbjob.keys()) - 1:
                sb_header += "#SBATCH hetjob\n"
            for job_name, details in job_details.items():
                # For one hetgroup, we don't need to specify hetgroup option
                het_group_id = het_group
                if len(sbjob) == 1:
                    het_group_id = -1
                sb_steps += self.get_jobstep_cmd(
                    node,
                    job_name,
                    details,
                    het_group=het_group_id,
                    with_output=True,
                    as_step=True,
                )
        sb_script += sb_header + sb_steps + "wait"
        self.sb_script = sb_script

    def generate(self, node, job_name, details, node_features):
        sb_script = "#!/bin/bash\n"
        sb_script += f"#SBATCH --partition={SLURM_PARTITION}" + "\n"
        sb_script += f"#SBATCH --job-name={job_name}" + "\n"
        output = self.get_output_filepath(node, job_name)
        sb_script += f"#SBATCH --output={output}" + "\n"
        sb_script += f"#SBATCH --comment={details['comment']}" + "\n"

        n_cores = self.get_n_cores(details)

        if node_features is None:
            sb_script += f"#SBATCH --nodelist={node} -c {n_cores}" + "\n"
        else:
            sb_script += f"#SBATCH --constraint={job_name} -c {n_cores}" + "\n"

        sb_script += self.get_jobstep_cmd(node, job_name, details)
        self.sb_script = sb_script
        if self.verbose:
            print(self.sb_script)
            print("")
