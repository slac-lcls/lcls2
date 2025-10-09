import inspect
import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
from psana.psexp.tools import mode

def get_class_name(obj):
    """
    Returns the class name of the given object instance.
    """
    try:
        return obj.__class__.__name__
    except AttributeError:
        return str(type(obj))  # fallback for non-class cases


def get_logger(level="INFO", logfile=None, name=None, timestamp=False):
    if mode == "mpi":
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0

    # Add .rankX suffix if logfile is defined and running in MPI
    if logfile and rank is not None:
        path = Path(logfile)
        logfile = str(path.with_name(f"{path.stem}.rank{rank}{path.suffix}"))

    return Logger(
        name=name,
        level=level,
        myrank=rank,
        logfile=logfile,
        timestamp=timestamp
    )


class Logger:
    def __init__(self, name=None, level=logging.INFO, myrank=None, timestamp=False,
                 logfile=None, max_bytes=10*1024*1024, backup_count=3):

        if name is None:
            frame = inspect.stack()[1]
            module = inspect.getmodule(frame[0])
            name = module.__name__ if module else '__main__'

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        rank_str = f" RANK:{myrank}" if myrank is not None else ""
        time_str = "[%(asctime)s] " if timestamp else ""
        formatter = logging.Formatter(
            f'{time_str}[PSANA-%(levelname)s{rank_str}] %(name)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            self.logger.addHandler(stream_handler)

        # File handler (optional)
        if logfile:
            file_handler = RotatingFileHandler(
                logfile, maxBytes=max_bytes, backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            if not any(isinstance(h, RotatingFileHandler) for h in self.logger.handlers):
                self.logger.addHandler(file_handler)

    def debug(self, msg, *args, **kwargs):
        timeline = int(os.environ.get("PS_TIMELINE", "0"))
        if "TIMELINE" in msg and not timeline:
            return
        self.logger.debug(msg, *args, **kwargs)
    def info(self, msg, *args, **kwargs): self.logger.info(msg, *args, **kwargs)
    def warning(self, msg, *args, **kwargs): self.logger.warning(msg, *args, **kwargs)
    def error(self, msg, *args, **kwargs): self.logger.error(msg, *args, **kwargs)
    def critical(self, msg, *args, **kwargs): self.logger.critical(msg, *args, **kwargs)

class WeakList(list):
    """Wrapper to make a list weak referenceable."""
    ...
class WeakDict(dict):
    """Wrapper to make a dict weak referenceable."""
    ...

def make_weak_refable(d):
    """Return a weak-referenceable wrapper of a dictionary.

    Used, e.g., for calibration constants.
    """
    new_d = WeakDict({})
    if d is None:
        return new_d
    for key in d:
        if isinstance(d[key], dict):
            new_d[key] = make_weak_refable(WeakDict(d[key]))
        elif isinstance(d[key], tuple):
            new_d[key] = WeakList(list(d[key]))
        elif d[key] is None:
            new_d[key] = WeakDict({})
        else:
            new_d[key] = d[key]
    return new_d
