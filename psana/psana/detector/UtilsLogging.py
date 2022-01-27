
"""
Usage::
        from psana.detector.UtilsLogging import logging, DICT_NAME_TO_LEVEL, STR_LEVEL_NAMES, init_logger
        logger = logging.getLogger(__name__)
        init_logger(loglevel='DEBUG', logfname=None)
"""

import os
import sys
import logging
DICT_NAME_TO_LEVEL = logging._nameToLevel
STR_LEVEL_NAMES = ', '.join(DICT_NAME_TO_LEVEL.keys())

##logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(filename)s: %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)
##logging.basicConfig(filename='log.txt', filemode='w', format=fmt, level=DICT_NAME_TO_LEVEL[args.logmode])

def logger_formatter_int_loglevel(loglevel='DEBUG'):
    int_loglevel = DICT_NAME_TO_LEVEL[loglevel.upper()]
    logger = logging.getLogger()
    logger.setLevel(int_loglevel) # logging.DEBUG
    fmt = '[%(levelname).1s] %(filename)s L%(lineno)04d %(message)s' if int_loglevel==logging.DEBUG else\
          '[%(levelname).1s] L%(lineno)04d %(message)s' # %(asctime)s
    formatter = logging.Formatter(fmt)
    return logger, logging.Formatter(fmt), int_loglevel

def init_stream_handler(loglevel='DEBUG'):
    logger, formatter, int_loglevel = logger_formatter_int_loglevel(loglevel)
    strh = logging.StreamHandler(sys.stdout)
    strh.setLevel(int_loglevel)
    strh.setFormatter(formatter)
    logger.addHandler(strh)

def init_file_handler(loglevel='DEBUG', logfname=None, filemode=0o664):
    if logfname is None: return
    logger, formatter, int_loglevel = logger_formatter_int_loglevel(loglevel)
    filh = logging.FileHandler(logfname)
    filh.setLevel(int_loglevel) # logging.DEBUG
    filh.setFormatter(formatter)
    logger.addHandler(filh)
    os.chmod(logfname, filemode)

def init_logger(loglevel='DEBUG', logfname=None, filemode=0o664):
    init_stream_handler(loglevel)
    init_file_handler(loglevel, logfname, filemode)

# EOF
