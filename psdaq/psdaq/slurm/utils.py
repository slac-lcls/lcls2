import os
from dataclasses import dataclass, field
import asyncio
import psutil
from datetime import datetime

import socket
LOCALHOST = socket.gethostname()

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

    def get_jobstep_cmd(self, job_name, details, het_group=-1, output=None):
        output_opt = ''
        if output:
            output_opt = f"--output={output} "
        env_opt = ''
        if details['env'] != '':
            env = details['env'].replace(" ", ",").strip("'")
            env_opt = f"--export=ALL,{env} "
        het_group_opt = ''
        if het_group > -1:
            het_group_opt = f"--het-group={het_group} "
            
        cmd = details['cmd']
        if details['conda_env'] != '':
            CONDA_EXE = os.environ.get('CONDA_EXE','')
            loc_inst = CONDA_EXE.find('/inst')
            conda_profile = os.path.join(CONDA_EXE[:loc_inst],'inst','etc','profile.d','conda.sh')
            cmd = f"source {conda_profile}; conda activate {details['conda_env']}; {cmd}"
        
        jobstep_cmd = f"srun -n1 --exclusive --job-name={job_name} {het_group_opt}{output_opt}{env_opt} bash -c '{cmd}'" + "& \n"
        return jobstep_cmd


    def generate_as_step(self, sbjob):
        sb_script = "#!/bin/bash\n"
        sb_script += f"#SBATCH --partition=anaq"+"\n"
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
                sb_steps += self.get_jobstep_cmd(job_name, details, het_group=het_group, output=output)
        sb_script += sb_header + sb_steps + "wait"
        self.sb_script = sb_script

    def generate(self, node, job_name, details):
        sb_script = "#!/bin/bash\n"
        sb_script += f"#SBATCH --partition=anaq"+"\n"
        sb_script += f"#SBATCH --job-name={job_name}"+"\n"
        
        if node == "localhost": node = LOCALHOST
        sb_script += f"#SBATCH --nodelist={node} --ntasks=1"+"\n"
        
        output = self.get_output_filepath(node, job_name)
        sb_script += f"#SBATCH --output={output}"+"\n"
        
        sb_script += self.get_jobstep_cmd(job_name, details)
        sb_script += "wait"
        self.sb_script = sb_script

