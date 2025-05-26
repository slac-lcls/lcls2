import inspect
import logging
from logging.handlers import RotatingFileHandler


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

    def debug(self, msg, *args, **kwargs): self.logger.debug(msg, *args, **kwargs)
    def info(self, msg, *args, **kwargs): self.logger.info(msg, *args, **kwargs)
    def warning(self, msg, *args, **kwargs): self.logger.warning(msg, *args, **kwargs)
    def error(self, msg, *args, **kwargs): self.logger.error(msg, *args, **kwargs)
    def critical(self, msg, *args, **kwargs): self.logger.critical(msg, *args, **kwargs)
