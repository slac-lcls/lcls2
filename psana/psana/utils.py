import os


class Logger:
    def __init__(self, myrank=None):
        self.myrank = myrank
        self.verbosity = int(os.environ.get("PS_VERBOSITY", "0"))

    def log(self, msg, level=0):
        if level <= self.verbosity:
            if self.myrank is not None:
                print(f"rank:{self.myrank} {msg}", flush=True)
            else:
                print(f"{msg}", flush=True)

    def info(self, msg):
        self.log(msg, level=1)

    def debug(self, msg, level=2):
        self.log(msg, level=level)
