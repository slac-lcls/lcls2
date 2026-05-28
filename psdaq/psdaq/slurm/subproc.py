import asyncio
import sys


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

    async def run_exec(self, args, wait_output=False, env=None):
        """Start a subprocess without a shell."""
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        if wait_output:
            stdout, stderr = await proc.communicate()
            if stdout:
                print(stdout.decode(), end="")
            if stderr:
                print(stderr.decode(), end="", file=sys.stderr)
            return proc.returncode

        self.procs[proc.pid] = proc
        return proc.pid

    def pids(self):
        return list(self.procs.keys())
