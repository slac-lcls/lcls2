import asyncio
import psutil


class SubprocHelper:
    def __init__(self):
        self.procs = {}

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

    async def run(self, cmd, wait_output=False):
        """Start psplot as a subprocess"""
        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        if wait_output:
            stdout = await self.read_stdout(proc)
            await proc.wait()
        else:
            self.procs[proc.pid] = proc

    def pids(self):
        return list(self.procs.keys())
