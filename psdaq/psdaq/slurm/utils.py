import os
from dataclasses import dataclass, field
import asyncio
import psutil
from datetime import datetime
import subprocess

import socket

LOCALHOST = socket.gethostname()
SLURM_PARTITION = "drpq"
DRP_N_RSV_CORES = 4


class PSbatchSubCommand:
    START = 0
    STOP = 1
    RESTART = 2


class SbatchManager:
    def __init__(self):
        self.sb_script = ""
        now = datetime.now()
        self.output_prefix_datetime = now.strftime("%d_%H:%M:%S")
        self.output_path = os.path.join(
            os.environ.get("HOME", ""), now.strftime("%Y"), now.strftime("%m")
        )
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        envs = [
            "CONDA_PREFIX",
            "CONFIGDB_AUTH",
            "TESTRELDIR",
            "RDMAV_FORK_SAFE",
            "RDMAV_HUGEPAGES_SAFE",
            "OPENBLAS_NUM_THREADS",
            "PS_PARALLEL",
        ]
        self.env_dict = {key: os.environ.get(key, "") for key in envs}
        self.as_step = False
        self.verbose = False

    @property
    def git_describe(self):
        git_describe = None
        if "TESTRELDIR" in os.environ:
            git_describe = self.call_subprocess(
                "git", "-C", os.environ["TESTRELDIR"], "describe", "--dirty", "--tag"
            )
        return git_describe

    def set_attr(self, attr, val):
        setattr(self, attr, val)

    def call_subprocess(self, *args):
        cc = subprocess.run(args, capture_output=True)
        output = None
        if not cc.returncode:
            output = str(cc.stdout.strip(), "utf-8")
        return output

    def get_comment(self, xpm_id, platform, config_id):
        comment = f"x{xpm_id}_p{platform}_{config_id}"
        return comment

    def get_node_features(self):
        lines = self.call_subprocess("sinfo", "-N", "-h", "-o", '"%N %f"').splitlines()
        node_features = {}
        for line in lines:
            node, features = line.strip('"').split()
            node_features[node] = {feature: 0 for feature in features.split(",")}
        return node_features

    def get_job_info(self):
        """Returns formatted output from squeue by the current user"""
        user = os.environ.get("USER", "")
        if not user:
            print(f"Cannot list jobs for user. $USER variable is not set.")
        else:
            if self.as_step:
                format_string = "JobIDRaw,Comment,JobName,State,NodeList"
                lines = self.call_subprocess(
                    "sacct", "-h", f"--format={format_string}"
                ).splitlines()
            else:
                format_string = '"%i %k %j %T %R"'
                lines = self.call_subprocess(
                    "squeue", "-u", user, "-h", "-o", format_string
                ).splitlines()

        job_details = {}
        for i, job_info in enumerate(lines):
            cols = job_info.strip('"').split()
            success = True
            if len(cols) == 5:
                job_id, comment, job_name, state, nodelist = cols
            elif len(cols) > 5:
                job_id, comment, job_name, state = cols[:4]
                nodelist = " ".join(cols[5:])
            else:
                success = False
            if success:
                job_details[comment] = {
                    "job_id": job_id,
                    "job_name": job_name,
                    "state": state,
                    "nodelist": nodelist,
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
        if not job_name.startswith("ami") and not job_name in (
            "groupca",
            "prom2pvs",
            "control_gui",
            "xpmpva",
        ):
            cmd += f" -u {job_name}"
        if "flags" in details:
            if details["flags"].find("p") > -1:
                cmd += f" -p {repr(self.platform)}"
        if job_name.startswith("ami-meb"):
            cmd += f" -u {job_name}"
        if self.is_drp(details["cmd"]):
            n_workers = self.get_n_cores(details) - DRP_N_RSV_CORES
            if n_workers < 1:
                n_workers = 1
            cmd += f" -W {n_workers}"
        return cmd

    def is_drp(self, cmd):
        return cmd.strip().startswith("drp ")

    def get_output_header(self, node, job_name, details):
        header = ""
        header += "# ID:      %s\n" % job_name
        header += "# PLATFORM:%s\n" % self.platform
        header += "# HOST:    %s\n" % node
        header += "# CMDLINE: %s\n" % self.get_daq_cmd(details, job_name)
        # obfuscating the password in the log
        clear_auth = self.env_dict["CONFIGDB_AUTH"]
        self.env_dict["CONFIGDB_AUTH"] = "*****"
        for key2, val2 in self.env_dict.items():
            header += f"# {key2}:{val2}\n"
        self.env_dict["CONFIGDB_AUTH"] = clear_auth
        if "TESTRELDIR" in self.env_dict:
            git_describe = self.git_describe
            if git_describe:
                header += "# GIT_DESCRIBE:%s\n" % git_describe
        return header

    def get_n_cores(self, details):
        n_cores = 1
        if "cores" in details:
            n_cores = int(details["cores"])
        return n_cores

    def get_jobstep_cmd(self, node, job_name, details, het_group=-1, with_output=False):
        output_opt = ""
        if with_output:
            output = self.get_output_filepath(node, job_name)
            output_opt = f"--output={output} --open-mode=append "
        env_opt = "--export=ALL"
        if "env" in details:
            if details["env"] != "":
                env = details["env"].replace(" ", ",").strip("'")
                env_opt += f",{env}"
        het_group_opt = ""
        if het_group > -1:
            het_group_opt = f"--het-group={het_group} "

        cmd = self.get_daq_cmd(details, job_name)
        header = self.get_output_header(node, job_name, details)
        cmd = f'echo "{header}"; {cmd}'
        if "conda_env" in details:
            if details["conda_env"] != "":
                CONDA_EXE = os.environ.get("CONDA_EXE", "")
                loc_inst = CONDA_EXE.find("/inst")
                conda_profile = os.path.join(
                    CONDA_EXE[:loc_inst], "inst", "etc", "profile.d", "conda.sh"
                )
                cmd = f"source {conda_profile}; conda activate {details['conda_env']}; {cmd}"

        x11_opt = ""
        if "flags" in details:
            if details["flags"].find("x") > -1:
                x11_opt = f"--x11 xterm -e "
                env_opt += ",DISPLAY=localhost:11.0"

        env_opt += " "
        n_cores = self.get_n_cores(details)
        jobstep_cmd = f"srun -n1 --cpus-per-task={n_cores} --exclusive --job-name={job_name} {het_group_opt}{output_opt}{env_opt} {x11_opt}bash -c '{cmd}'"

        if het_group > -1:
            jobstep_cmd += "&"

        jobstep_cmd += " \n"
        return jobstep_cmd

    def generate_as_step(self, sbjob, node_features):
        sb_script = "#!/bin/bash\n"
        sb_script += f"#SBATCH --partition={SLURM_PARTITION}" + "\n"
        sb_script += f"#SBATCH --job-name=main" + "\n"
        output = self.get_output_filepath(LOCALHOST, "slurm")
        sb_script += f"#SBATCH --output={output}" + "\n"
        sb_script += f"#SBATCH --output={output}" + "\n"
        sb_header = ""
        sb_steps = ""
        for het_group, (node, job_details) in enumerate(sbjob.items()):
            if node == "localhost" or node_features is None:
                if node == "localhost":
                    node = LOCALHOST
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
                    node, job_name, details, het_group=het_group_id, with_output=True
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

        if node == "localhost" or node_features is None:
            if node == "localhost":
                node = LOCALHOST
            sb_script += f"#SBATCH --nodelist={node} --ntasks={n_cores}" + "\n"
        else:
            sb_script += f"#SBATCH --constraint={job_name} --ntasks={n_cores}" + "\n"

        sb_script += self.get_jobstep_cmd(node, job_name, details)
        sb_script += "wait"
        self.sb_script = sb_script
        if self.verbose:
            print(self.sb_script)
            print("")
