import os, sys
from dataclasses import dataclass, field
import asyncio
import psutil
from datetime import datetime
import subprocess

import socket

LOCALHOST = socket.gethostname()
SLURM_PARTITION = "drpq"
DRP_N_RSV_CORES = int(os.environ.get('PS_DRP_N_RSV_CORES', '4'))
SCRIPTS_ROOTDIR = '/reg/g/pcds/dist/pds'


class PSbatchSubCommand:
    START = 0
    STOP = 1
    RESTART = 2

def call_subprocess(*args):
    cc = subprocess.run(args, capture_output=True)
    output = None
    if not cc.returncode:
        output = str(cc.stdout.strip(), "utf-8")
    return output

class SbatchManager:
    def __init__(self, configfilename, platform, as_step, verbose, output=None):
        self.sb_script = ""
        now = datetime.now()
        self.output_prefix_datetime = now.strftime("%d_%H:%M:%S")
        if output is None:
            self.output_path = os.path.join(
                os.environ.get("HOME", ""), now.strftime("%Y"), now.strftime("%m")
            )
        else:
            self.output_path = output
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.configfilename = configfilename
        self.platform = platform
        self.as_step = as_step
        self.verbose = verbose
        self.user = os.environ['USER'] 
        self.hutch = self.user[:self.user.find('opr')]
        self.scripts_dir = os.path.join(SCRIPTS_ROOTDIR, self.hutch, 'scripts')

    def set_attr(self, attr, val):
        setattr(self, attr, val)

    def get_comment(self, xpm_id, platform, config_id):
        comment = f"x{xpm_id}_p{platform}_{config_id}"
        return comment

    def get_node_features(self):
        lines = call_subprocess("sinfo", "-N", "-h", "-o", '"%N %f"').splitlines()
        node_features = {}
        for line in lines:
            node, features = line.strip('"').split()
            node_features[node] = {feature: 0 for feature in features.split(",")}
        return node_features

    def get_job_info_byid(self, job_id, jobparms):
        """Returns a dictionary containing values obtained from scontrol
        with the given jobparms list. Returns {} if this job does not exist.
        """
        output = call_subprocess(
            "scontrol", "show", "job", job_id
        )
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
        """Returns formatted output from squeue by the current user"""
        user = self.user 
        if not user:
            print(f"Cannot list jobs for user. $USER variable is not set.")
        else:
            if use_sacct:
                format_string = "JobIDRaw,Comment,JobName,State,NodeList"
                lines = call_subprocess(
                    "sacct", "-u", user, "-n", f"--format={format_string}"
                ).splitlines()
            else:
                format_string = '"%i %k %j %T %R"'
                lines = call_subprocess(
                    "squeue", "-u", user, "-h", "-o", format_string
                ).splitlines()

        job_details = {}
        for i, job_info in enumerate(lines):
            cols = job_info.strip('"').split()
            # Check that JobId column has all the characters as digit
            if not cols[0].isdigit(): continue

            success = True
            if len(cols) == 5:
                job_id, comment, job_name, state, nodelist = cols
            elif len(cols) > 5:
                job_id, comment, job_name, state = cols[:4]
                nodelist = " ".join(cols[5:])
            else:
                success = False
            
            if success:
                # Get logfile from job_id
                scontrol_result = call_subprocess(
                    "scontrol", "show", "job", job_id
                )
                logfile = ""
                if scontrol_result is not None:
                    scontrol_lines = scontrol_result.splitlines()
                    for scontrol_line in scontrol_lines:
                        if scontrol_line.find("StdOut") > -1:
                            scontrol_cols = scontrol_line.split("=")
                            logfile = scontrol_cols[1]
                
                # Results from sacct also show old jobs with the same name.
                # We choose the oldest job and returns its values. 
                if comment not in job_details:
                    job_details[comment] = {
                        "job_id": job_id,
                        "job_name": job_name,
                        "state": state,
                        "nodelist": nodelist,
                        "logfile": logfile,
                    }
                elif int(job_id) > int(job_details[comment]["job_id"]):
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
        if not job_name.startswith("ami") and not job_name in (
            "groupca",
            "prom2pvs",
            "control_gui",
            "xpmpva",
            "psqueue",
        ):
            cmd += f" -u {job_name}"
        if "flags" in details:
            if details["flags"].find("p") > -1:
                cmd += f" -p {repr(self.platform)}"
        if job_name.startswith("ami-meb"):
            cmd += f" -u {job_name}"
        if job_name == "psqueue":
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

    def get_jobstep_cmd(
        self, node, job_name, details, het_group=-1, with_output=False, as_step=False
    ):
        output_opt = ""
        if with_output:
            output = self.get_output_filepath(node, job_name)
            output_opt = f"--output={output} --open-mode=append "

        env_opt = "--export="

        # Inherit follows from user's account
        env_opt +="HOME"
        env_opt +=",USER"
        env_opt +=",TESTRELDIR"
        env_opt +=",CONDA_PREFIX"
        env_opt +=",CONDA_DEFAULT_ENV"
        env_opt +=",CONDA_EXE"
        env_opt +=",CONFIGDB_AUTH"
        
        # Build PATH and PYTHONPATH from scratch
        daq_path ="$TESTRELDIR/bin"
        daq_path+=":$CONDA_PREFIX/bin"
        daq_path+=":$CONDA_PREFIX/epics/bin/linux-x86_64"
        daq_path+=":/usr/sbin"
        daq_path+=":/usr/bin"
        daq_path+=":/sbin"
        daq_path+=":/bin"
        env_opt +=",PATH="+daq_path
        env_opt +=f",PYTHONPATH=$TESTRELDIR/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"

        # For x11 forwarding
        env_opt +=",DISPLAY"
        env_opt +=",XAUTHORITY=$HOME/.Xauthority"
        
        # Include any exists in setup_env.sh backdoor
        env_opt +=",$DAQBATCH_EXPORT"
        
        # Include any exists in the configuration file
        cnf_env = ""
        found_ld_library_path = False
        if "env" in details:
            if details["env"] != "":
                envs = details["env"].split()
                for i, env in enumerate(envs):
                    env_name, env_var = env.split('=')
                    if env_name == "LD_LIBRARY_PATH":
                        found_ld_library_path = True
                    cnf_env += ','+env 
        env_opt += cnf_env 
        
        if not found_ld_library_path:
            env_opt +=",LD_LIBRARY_PATH=$TESTRELDIR/lib"

        env_opt += " "
        
        het_group_opt = ""
        if het_group > -1:
            het_group_opt = f"--het-group={het_group} "

        cmd = self.get_daq_cmd(details, job_name)
        daqlog_header = f'daqlog_header {job_name} {self.platform} {node} "{cmd.strip()}"'
        cmd = f'{daqlog_header}; {cmd}'
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
        sb_script += f"#SBATCH --job-name=main" + "\n"
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
        n_tasks = 1

        if node_features is None:
            sb_script += f"#SBATCH --nodelist={node} -c {n_cores}" + "\n"
        else:
            sb_script += f"#SBATCH --constraint={job_name} -c {n_cores}" + "\n"

        sb_script += self.get_jobstep_cmd(node, job_name, details)
        self.sb_script = sb_script
        if self.verbose:
            print(self.sb_script)
            print("")
