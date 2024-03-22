import os
from dataclasses import dataclass, field
import asyncio
import psutil
import socket
LOCALHOST = socket.gethostname()

class PSbatchSubCommand:
    START=0
    STOP=1
    RESTART=2

class SbatchManager:
    def __init__(self):
        self.sb_script = ''

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
    
    def generate_as_step(self, sbjob):
        sb_script = "#!/bin/bash\n"
        sb_script += f"#SBATCH --partition=anaq"+"\n"
        sb_script += f"#SBATCH --job-name=main"+"\n"
        sb_header = ''
        sb_steps = '' 
        for het_group, (node, job_details) in enumerate(sbjob.items()):
            if node == "localhost": node = LOCALHOST
            sb_header += f"#SBATCH --nodelist={node} --ntasks={len(job_details.keys())}" + "\n"
            if het_group < len(sbjob.keys()) - 1:
                sb_header += "#SBATCH hetjob\n"
            for job_name, details in job_details.items():
                sb_steps += f"srun -n1 --job-name={job_name} --het-group={het_group} --exclusive {details['cmd']}" + "& \n"
        sb_script += sb_header + sb_steps + "wait"
        self.sb_script = sb_script
        print(self.sb_script)

    def generate(self, node, job_name, details):
        sb_script = "#!/bin/bash\n"
        sb_script += f"#SBATCH --partition=anaq"+"\n"
        sb_script += f"#SBATCH --job-name={job_name}"+"\n"
        if node == "localhost": node = LOCALHOST
        sb_script += f"#SBATCH --nodelist={node} --ntasks=1"+"\n"
        sb_script += f"srun -n1 " + details['cmd'] + "\n"
        self.sb_script = sb_script

