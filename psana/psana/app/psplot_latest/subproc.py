import asyncio
import psutil

class SubprocHelper():
    def __init__(self):
        self.procs = {}

    async def _run(self, cmd, prv='usr'):
        """ Start psplot as a subprocess """
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        self.procs[proc.pid] = proc

    def pids(self):
        return list(self.procs.keys())






