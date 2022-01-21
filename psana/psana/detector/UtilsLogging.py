
"""
Usage::
        from psana.detector.UtilsLogging import logging, DICT_NAME_TO_LEVEL, STR_LEVEL_NAMES, init_logger
        logger = logging.getLogger(__name__)
        init_logger(loglevel='DEBUG', logfname=None)
"""

import logging
logger = logging.getLogger(__name__)
#DICT_NAME_TO_LEVEL = {k:v for k,v in logging._levelNames.iteritems() if isinstance(k, str)} # py2
DICT_NAME_TO_LEVEL = logging._nameToLevel
STR_LEVEL_NAMES = ', '.join(DICT_NAME_TO_LEVEL.keys())

#fmt = '[%(levelname).1s] %(name)s %(message)s' if args.logmode=='DEBUG' else '[%(levelname).1s] %(message)s'
#logging.basicConfig(format=fmt, level=DICT_NAME_TO_LEVEL[args.logmode])
##logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(filename)s: %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)
##logging.basicConfig(filename='log.txt', filemode='w', format=fmt, level=DICT_NAME_TO_LEVEL[args.logmode])
#logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s %(message)s', level=logging.INFO)
#logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(message)s', level=logging.INFO)

def init_logger(loglevel='DEBUG', logfname=None):
    import sys

    int_loglevel = DICT_NAME_TO_LEVEL[loglevel.upper()]

    logger = logging.getLogger()
    logger.setLevel(int_loglevel) # logging.DEBUG
    fmt = '[%(levelname).1s] %(filename)s L%(lineno)04d %(message)s' if int_loglevel==logging.DEBUG else\
          '[%(levelname).1s] L%(lineno)04d %(message)s'
    formatter = logging.Formatter(fmt)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(int_loglevel)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    if logfname is not None:
        file_handler = logging.FileHandler(logfname) #'log-in-file-test.log'
        file_handler.setLevel(int_loglevel) # logging.DEBUG
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

# EOF
