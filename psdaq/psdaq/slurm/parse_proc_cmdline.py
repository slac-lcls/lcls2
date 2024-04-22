import asyncio
import sys

async def read_stdout(proc):
    # Read data from stdout until EOF
    data = ""
    while True:
        line = await proc.stdout.readline()
        if line:
            data += line.decode()
        else:
            break
    return data

async def run(pid):
    cmd = f"cat /proc/{pid}/cmdline | strings -1"
    proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
            )
    stdout = await read_stdout(proc)
    print(' '.join(stdout.split()))
    await proc.wait()

if __name__ == "__main__":
    pid = sys.argv[1]
    asyncio.run(run(pid))

