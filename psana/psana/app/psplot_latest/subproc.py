import asyncio
import psutil

class SubprocHelper():
    async def _run(self, cmd, prv='usr'):
        """ Start psplot as a subprocess """
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)

    def terminate(self, pid, timeout=3):
        p = psutil.Process(pid)
        print(f'Terminating process: {pid}...')
        p.terminate()
        p.wait(timeout=timeout)
        if psutil.pid_exists(pid):
            print(f'Failed, could not terminate the process.')
        else:
            print(f'Succeeded')





