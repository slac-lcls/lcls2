import inspect
import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
from psana.psexp.tools import mode

# --- new: one-time config state ---
_LOGGING_CONFIGURED = False
_DEFAULT_LEVEL = "INFO"
_DEFAULT_LOGFILE = None
_DEFAULT_TIMESTAMP = False

def configure_logging(level="INFO", logfile=None, timestamp=False, *, force=False):
    """
    Configure default logging for psana once. Safe to call multiple times.
    If 'force' is True, existing psana loggers get their handlers replaced.
    """
    global _LOGGING_CONFIGURED, _DEFAULT_LEVEL, _DEFAULT_LOGFILE, _DEFAULT_TIMESTAMP
    _DEFAULT_LEVEL = level
    _DEFAULT_LOGFILE = logfile
    _DEFAULT_TIMESTAMP = timestamp
    _LOGGING_CONFIGURED = True

    # Optionally update existing psana loggers
    if force:
        # Rebuild handlers for all known psana loggers
        logging.getLogger()  # scan manager
        for name, logger in logging.root.manager.loggerDict.items():
            if not isinstance(logger, logging.Logger):
                continue
            if not name.startswith("psana"):
                continue
            _rebuild_handlers(logger, level=level, logfile=logfile, timestamp=timestamp)

def _rebuild_handlers(py_logger, level, logfile, timestamp):
    """Replace handlers on an existing logging.Logger with our format/targets."""
    # Remove existing handlers
    for h in list(py_logger.handlers):
        py_logger.removeHandler(h)

    # Attach fresh handlers
    rank = _get_rank()
    formatter = _make_formatter(timestamp, rank)
    _attach_handlers(py_logger, formatter, logfile, rank)
    py_logger.setLevel(level)
    py_logger.propagate = False  # avoid double-logging via root

def _get_rank():
    if mode == "mpi":
        try:
            from mpi4py import MPI
            return MPI.COMM_WORLD.Get_rank()
        except Exception:
            return 0
    return 0

def _make_formatter(timestamp, rank):
    rank_str = f" RANK:{rank}" if rank is not None else ""
    time_str = "[%(asctime)s] " if timestamp else ""
    return logging.Formatter(
        f'{time_str}[PSANA-%(levelname)s{rank_str}] %(name)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def _attach_handlers(py_logger, formatter, logfile, rank):
    # Console
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    py_logger.addHandler(sh)

    # File (optional)
    if logfile:
        path = Path(logfile)
        # Suffix with rank if in MPI
        if rank is not None:
            logfile = str(path.with_name(f"{path.stem}.rank{rank}{path.suffix}"))
            path = Path(logfile)

        # Ensure directory exists
        if path.parent and str(path.parent) not in (".", ""):
            os.makedirs(path.parent, exist_ok=True)

        fh = RotatingFileHandler(logfile, maxBytes=10*1024*1024, backupCount=3)
        fh.setFormatter(formatter)
        py_logger.addHandler(fh)

def get_class_name(obj):
    try:
        return obj.__class__.__name__
    except AttributeError:
        return str(type(obj))

def get_logger(level=None, logfile=None, name=None, timestamp=False):
    """
    Backward-compatible entry point that returns your Logger wrapper.
    If configure_logging(...) was called, defaults come from there.
    """
    # Use configured defaults unless explicit values were passed
    lvl = level if level is not None else (_DEFAULT_LEVEL if _LOGGING_CONFIGURED else "INFO")
    logf = logfile if logfile is not None else (_DEFAULT_LOGFILE if _LOGGING_CONFIGURED else None)
    ts  = timestamp if timestamp is not False else (_DEFAULT_TIMESTAMP if _LOGGING_CONFIGURED else False)
    return Logger(name=name, level=lvl, myrank=_get_rank(), logfile=logf, timestamp=ts)

class Logger:
    def __init__(self, name=None, level=logging.INFO, myrank=None, timestamp=False,
                 logfile=None, max_bytes=10*1024*1024, backup_count=3):

        if name is None:
            # keep your existing behavior
            frame = inspect.stack()[1]
            module = inspect.getmodule(frame[0])
            name = module.__name__ if module else '__main__'

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Prevent duplicate emission via root
        self.logger.propagate = False

        rank_str = f" RANK:{myrank}" if myrank is not None else ""
        time_str = "[%(asctime)s] " if timestamp else ""

        class EpochNanoFormatter(logging.Formatter):
            def format(self, record):
                created_ns = int(record.created * 1_000_000_000)
                sec = created_ns // 1_000_000_000
                nsec = created_ns % 1_000_000_000
                record.ps_time = f"{sec}.{nsec:09d}"
                return super().format(record)

        formatter = EpochNanoFormatter(
            f'{time_str}[PSANA-%(levelname)s{rank_str} %(ps_time)s] %(name)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler (ensure only one StreamHandler on THIS logger)
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

        # File handler (ensure only one RotatingFileHandler on THIS logger)
        if logfile and not any(isinstance(h, RotatingFileHandler) for h in self.logger.handlers):
            path = Path(logfile)
            if myrank is not None:
                logfile = str(path.with_name(f"{path.stem}.rank{myrank}{path.suffix}"))
                path = Path(logfile)
            if path.parent and str(path.parent) not in (".", ""):
                os.makedirs(path.parent, exist_ok=True)

            file_handler = RotatingFileHandler(logfile, maxBytes=max_bytes, backupCount=backup_count)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def debug(self, msg, *args, **kwargs):
        if "TIMELINE" in msg and not int(os.environ.get("PS_TIMELINE", "0")):
            return
        self.logger.debug(msg, *args, **kwargs)
    def info(self, msg, *args, **kwargs): self.logger.info(msg, *args, **kwargs)
    def warning(self, msg, *args, **kwargs): self.logger.warning(msg, *args, **kwargs)
    def error(self, msg, *args, **kwargs): self.logger.error(msg, *args, **kwargs)
    def critical(self, msg, *args, **kwargs): self.logger.critical(msg, *args, **kwargs)


def first_timestamp(dgrams):
    """
    Return the first non-None timestamp found in a sequence of dgrams.

    Accepts either a callable .timestamp() or an attribute .timestamp.
    Raises RuntimeError if no usable timestamp is found.
    """
    for dg in dgrams:
        if not dg:
            continue
        try:
            ts_attr = getattr(dg, "timestamp", None)
            ts = ts_attr() if callable(ts_attr) else ts_attr
            if ts is not None:
                return ts
        except Exception:
            # Keep scanning the rest
            continue
    raise RuntimeError("No valid timestamp found in provided dgrams")

def first_service(dgrams):
    """
    Return the service code (1..13) from the first valid dgram in `dgrams`.

    Extracts via (d.env() >> 24) & 0xF, like Event.service().
    Raises RuntimeError if no valid service is found or if the extracted
    value is 0/out of the expected range.
    """
    for d in dgrams:
        if not d:
            continue
        try:
            svc = (d.env() >> 24) & 0xF
        except Exception:
            # If this dgram can't provide env(), skip it.
            continue
        if not (1 <= svc <= 13):
            raise RuntimeError(f"expected value between 1-13, got: {svc}")
        return svc

    raise RuntimeError("No valid service found in provided dgrams")

def first_env(dgrams):
    """
    Return d.env() from the first valid dgram in `dgrams`.
    Raises RuntimeError if none of the dgrams can provide env().
    """
    for d in dgrams:
        if not d:
            continue
        try:
            return d.env()
        except Exception:
            # If this dgram can't provide env(), skip it.
            continue
    raise RuntimeError("No valid dgram with env() found")
