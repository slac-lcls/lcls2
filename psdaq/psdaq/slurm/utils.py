import os
from dataclasses import dataclass, field
import asyncio
import psutil

class PSbatchSubCommand:
    START=0
    STOP=1
    RESTART=2

@dataclass
class SbatchParms:
    partition: str
    nodelist: list[str] 
    ntasks: int
    time: str = "00:05:00"
    job_name: str = "psbatch"

class SbatchManager:
    def __init__(self):
        self.job_file = "_sbatch.sh"

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
    
    def create_job_file(self, sbparms, cmd):
        with open(self.job_file, "w") as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines(f"#SBATCH --partition={sbparms.partition}"+"\n")
            fh.writelines(f"#SBATCH --job-name={sbparms.job_name}"+"\n")
            fh.writelines(f"#SBATCH --nodelist={','.join(sbparms.nodelist)}"+"\n")
            fh.writelines(f"#SBATCH --ntasks={sbparms.ntasks}"+"\n")
            fh.writelines(f"#SBATCH --time={sbparms.time}"+"\n")
            fh.writelines(f"t_start=`date +%s`"+"\n")
            fh.writelines(f"set -xe"+"\n")
            fh.writelines(cmd+"\n")
            fh.writelines("echo $SLURM_JOB_ID\n")
            fh.writelines(f"t_end=`date +%s`"+"\n")
            fh.writelines(f"echo PSJobCompleted TotalElapsed $((t_end-t_start))")

        
