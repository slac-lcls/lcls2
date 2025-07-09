import asyncio


class SubprocHelper:
    def __init__(self):
        self.procs = {}

    async def _run(self, cmd, callback=None):
        """Start psplot as a subprocess"""
        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        self.procs[proc.pid] = proc
        if callback:
            callback(proc.pid)

    def pids(self):
        return list(self.procs.keys())
