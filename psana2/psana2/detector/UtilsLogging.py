
"""
Usage::
        from psana2.detector.UtilsLogging import logging, DICT_NAME_TO_LEVEL, STR_LEVEL_NAMES, init_logger
        logger = logging.getLogger(__name__)
        init_logger(loglevel='DEBUG', logfname=None)
"""

import os
import sys
import logging
DICT_NAME_TO_LEVEL = logging._nameToLevel
STR_LEVEL_NAMES = ', '.join(DICT_NAME_TO_LEVEL.keys())
FMTDEF = '[%(levelname).1s] %(filename)s L%(lineno)04d %(message)s'


def basic_ligging(**kwa):
    """kwa:
       format=FMTDEF, datefmt='%H:%M:%S', level=logging.DEBUG
       filename='log.txt', filemode='w', level=DICT_NAME_TO_LEVEL[logmode]
    """
    logging.basicConfig(**kwa)


def logger_formatter_int_loglevel(loglevel='DEBUG', fmt=FMTDEF):
    global logger
    int_loglevel = DICT_NAME_TO_LEVEL[loglevel.upper()]
    logger = logging.getLogger()
    logger.setLevel(int_loglevel) # logging.DEBUG
    formatter = logging.Formatter(fmt)
    return logger, logging.Formatter(fmt), int_loglevel


def init_stream_handler(loglevel='DEBUG', fmt=FMTDEF):
    logger, formatter, int_loglevel = logger_formatter_int_loglevel(loglevel, fmt)
    strh = logging.StreamHandler(sys.stdout)
    strh.setLevel(int_loglevel)
    strh.setFormatter(formatter)
    logger.addHandler(strh)
    logger.debug('%s\nCommand: %s' % ((50*'_'), ' '.join(sys.argv)))


def init_file_handler(loglevel='DEBUG', logfname=None, filemode=0o664, group='ps-users', fmt=FMTDEF, **kwa):
    logger, formatter, int_loglevel = logger_formatter_int_loglevel(loglevel, fmt)
    if logfname is None:
        logger.warning('logfname is None - loge file is not saved')
        return
    filh = logging.FileHandler(logfname)
    filh.setLevel(int_loglevel) # logging.DEBUG
    filh.setFormatter(formatter)
    logger.addHandler(filh)
    os.chmod(logfname, filemode)
import psana2.pyalgos.generic.Utils as gu
    gu.change_file_ownership(logfname, user=None, group='ps-users')


def init_logger(loglevel='DEBUG', logfname=None, filemode=0o664, fmt=FMTDEF):
    init_stream_handler(loglevel, fmt)
    init_file_handler(loglevel, logfname, filemode, fmt=fmt)

# EOF
