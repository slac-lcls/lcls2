import os
from dataclasses import dataclass, field
import asyncio
import psutil
from datetime import datetime
from subprocess import run

import socket
LOCALHOST = socket.gethostname()
SLURM_PARTITION = 'drpq'

class PSbatchSubCommand:
    START=0
    STOP=1
    RESTART=2

class SbatchManager:
    def __init__(self):
        self.sb_script = ''
        now = datetime.now()
        self.output_prefix_datetime = now.strftime('%d_%H:%M:%S') 
        self.output_path = os.path.join(os.environ.get('HOME',''),
                now.strftime('%Y'), now.strftime('%m'))
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        envs = ['CONDA_PREFIX', 'CONFIGDB_AUTH', 'TESTRELDIR', 'RDMAV_FORK_SAFE', 'RDMAV_HUGEPAGES_SAFE', 'OPENBLAS_NUM_THREADS',
                'PS_PARALLEL' ]
        self.env_dict = {key: os.environ.get(key, '') for key in envs}
    
    @property
    def git_describe(self):
        git_describe = None
        if 'TESTRELDIR' in os.environ:
            try:
                cc = run(["git", "-C", os.environ['TESTRELDIR'], "describe", "--dirty", "--tag"], capture_output=True)
                if not cc.returncode:
                    git_describe = str(cc.stdout.strip(), 'utf-8')
            except Exception:
                pass
        return git_describe

    def set_attr(self, attr, val):
        setattr(self, attr, val)

    async def read_stdout(self, proc):
        # Read data from stdout until EOF
        data = ""
        while True:
            line = await proc.stdout.readline()
            if line:
                data += line.decode()
            else:
                break
        return data
    
    async def run(self, sbatch_cmd, callback=None):
        proc = await asyncio.create_subprocess_shell(
                sbatch_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
                )
        stdout = await self.read_stdout(proc)
        if stdout.find('Submitted batch job') == 0:
            slurm_job_id = int(stdout.split()[-1])
            if callback:
                callback(slurm_job_id)
        await proc.wait()
    
    def get_output_filepath(self, node, job_name):
        output_filepath = os.path.join(self.output_path, 
                self.output_prefix_datetime + '_' + node + ':' + job_name + '.log')
        return output_filepath

    def get_daq_cmd(self, details, output, job_name):
        cmd = details['cmd']
        cmd += f' -p {repr(self.platform)} -u {job_name}'
        if 'x' in details['flags']:
            if output:
                cmd = f'xterm -l -lf {output} -e ' + cmd 
            else:
                cmd = 'xterm -e ' + cmd
        return cmd

    def write_output_header(self, output, node, job_name, details):
        with open(output, 'w') as f:
            f.write("# ID:      %s\n" % job_name)
            f.write("# PLATFORM:%s\n" % self.platform)
            f.write("# HOST:    %s\n" % node)
            f.write("# CMDLINE: %s\n" % details['cmd'])
            # obfuscating the password in the log
            clear_auth = self.env_dict['CONFIGDB_AUTH']
            self.env_dict["CONFIGDB_AUTH"] = "*****"
            for key2, val2 in self.env_dict.items():
                f.write(f"# {key2}:{val2}\n")
            self.env_dict["CONFIGDB_AUTH"] = clear_auth
            if 'TESTRELDIR' in self.env_dict:
                git_describe = self.git_describe
                if git_describe:
                    f.write("# GIT_DESCRIBE:%s\n" % git_describe)

    def get_jobstep_cmd(self, node, job_name, details, het_group=-1):
        output = self.get_output_filepath(node, job_name)
        output_opt = f"--output={output} --open-mode=append "
        env_opt = ''
        if 'env' in details:
            if details['env'] != '':
                env = details['env'].replace(" ", ",").strip("'")
                env_opt = f"--export=ALL,{env} "
        het_group_opt = ''
        if het_group > -1:
            het_group_opt = f"--het-group={het_group} "
            
        cmd = self.get_daq_cmd(details, output, job_name)
        if 'conda_env' in details:
            if details['conda_env'] != '':
                CONDA_EXE = os.environ.get('CONDA_EXE','')
                loc_inst = CONDA_EXE.find('/inst')
                conda_profile = os.path.join(CONDA_EXE[:loc_inst],'inst','etc','profile.d','conda.sh')
                cmd = f"source {conda_profile}; conda activate {details['conda_env']}; {cmd}"
        
        jobstep_cmd = f"srun -n1 --exclusive --job-name={job_name} {het_group_opt}{output_opt}{env_opt} bash -c '{cmd}'" + "& \n"
        return jobstep_cmd


    def generate_as_step(self, sbjob):
        sb_script = "#!/bin/bash\n"
        sb_script += f"#SBATCH --partition={SLURM_PARTITION}"+"\n"
        sb_script += f"#SBATCH --job-name=main"+"\n"
        output = self.get_output_filepath(LOCALHOST, "slurm")
        sb_script += f"#SBATCH --output={output}"+"\n"
        sb_header = ''
        sb_steps = '' 
        for het_group, (node, job_details) in enumerate(sbjob.items()):
            if node == "localhost": node = LOCALHOST
            sb_header += f"#SBATCH --nodelist={node} --ntasks={len(job_details.keys())}" + "\n"
            if het_group < len(sbjob.keys()) - 1:
                sb_header += "#SBATCH hetjob\n"
            for job_name, details in job_details.items():
                output = self.get_output_filepath(node, job_name)
                self.write_output_header(output, node, job_name, details)
                # For one hetgroup, we don't need to specify hetgroup option
                het_group_id = het_group
                if len(sbjob) == 1: het_group_id = -1
                sb_steps += self.get_jobstep_cmd(node, job_name, details, het_group=het_group_id)
        sb_script += sb_header + sb_steps + "wait"
        self.sb_script = sb_script

    def generate(self, node, job_name, details):
        sb_script = "#!/bin/bash\n"
        sb_script += f"#SBATCH --partition={SLURM_PARTITION}"+"\n"
        sb_script += f"#SBATCH --job-name={job_name}"+"\n"
        
        if node == "localhost": node = LOCALHOST
        sb_script += f"#SBATCH --nodelist={node} --ntasks=1"+"\n"
        
        sb_script += self.get_jobstep_cmd(node, job_name, details)
        sb_script += "wait"
        self.sb_script = sb_script

