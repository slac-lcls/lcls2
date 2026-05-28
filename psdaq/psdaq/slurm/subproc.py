import asyncio
import sys


class SubprocHelper:
    def __init__(self):
        self.procs = {}

    async def run_exec(self, args, wait_output=False, env=None, echo_output=True):
        """Start a subprocess without a shell."""
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        if wait_output:
            stdout, stderr = await proc.communicate()
            if echo_output or proc.returncode != 0:
                if stdout:
                    print(stdout.decode(), end="")
                if stderr:
                    print(stderr.decode(), end="", file=sys.stderr)
            return proc.returncode

        self.procs[proc.pid] = proc
        return proc.pid

    def pids(self):
        return list(self.procs.keys())
