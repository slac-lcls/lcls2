import subprocess

class SubprocHelper():
    def __init__(self):
        self.subprocs = {}

    def start_psplot(self, node, port, detname):
        """ Start psplot as a subprocess """
        subproc = subprocess.Popen(["psplot", "-s", node, "-p", str(port), detname])
        self.subprocs[subproc.pid] = subproc
        print(f"Start Process ID: {subproc.pid}")

    def terminate_psplot(self, pid):
        """ Terminate given process id """
        print(f"Terminating process id: {pid}")
        subproc = self.subprocs[pid]
        # Send SIGTER 
        subproc.terminate()
        # Wait for process to terminate
        returncode = subproc.wait()
        print(f"|-->Return code: {returncode}")
