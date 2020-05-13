#------------------------------
"""
:py:class:`logger` - set of utilities for standard python logging facility
==========================================================================

Usage ::

    #Test this module:  python lcls2/psana/psana/pyalgos/generic/logger.py

    from psana.pyalgos.generic.logger import logging, config_logger

    #loglevel is one of 'debug','info','warning','error','critical'
    config_logger(loglevel='info') #, filename='log.txt')

    logger = logging.getLogger('My_Module')

    logger.debug   ('This is a test message 1')
    logger.info    ('This is a test message 2')
    logger.warning ('This is a test message 3')
    logger.error   ('This is a test message 4')
    logger.critical('This is a test message 5')
    logger.exception(msg, *args, **kwargs)
    logger.log(level, msg, *args, **kwargs)

 See:
    - :py:class:`logger`
    - `matplotlib <https://docs.python.org/3/library/logging.html>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2018-03-15 by Mikhail Dubrovin
"""
#------------------------------

import logging

#LOGLEVELS = {'info'    : logging.INFO, 
#             'debug'   : logging.DEBUG,
#             'warning' : logging.WARNING,
#             'warn'    : logging.WARN,
#             'error'   : logging.ERROR,
#             'critical': logging.CRITICAL,
#             'notset'  : logging.NOTSET}

DICT_LEVEL_TO_NAME = logging._levelToName # {0: 'NOTSET', 50: 'CRITICAL',...
DICT_NAME_TO_LEVEL = logging._nameToLevel # {'INFO': 20, 'WARNING': 30, 'WARN': 30,...
LEVEL_NAMES = list(logging._levelToName.values())
STR_LEVEL_NAMES = ', '.join(DICT_NAME_TO_LEVEL.keys())
TSFORMAT = '%Y-%m-%dT%H:%M:%S' #%z'

#------------------------------

def init_logger(loglev_name='DEBUG', fmt='[%(levelname).1s] L%(lineno)04d : %(message)s', datefmt=TSFORMAT) :
    #fmt='%(asctime)s %(name)s %(lineno)d %(levelname)s: %(message)s' # '%(message)s'
    logging.basicConfig(format=fmt, datefmt=datefmt, level=DICT_NAME_TO_LEVEL[loglev_name])
    logging.debug('Logger is initialized for level %s' % loglev_name)

#------------------------------

def config_logger(loglevel='DEBUG',\
                  fmt='%(asctime)s %(name)s %(lineno)d %(levelname)s: %(message)s',\
                  datefmt=TSFORMAT,\
                  filename='',\
                  filemode='w') :

    level = DICT_NAME_TO_LEVEL.get(loglevel.upper(), logging.INFO)

    logging.basicConfig(format=fmt, datefmt=datefmt,\
                        level=level, filename=filename, filemode=filemode)

    #formatter = logging.Formatter(fmt, datefmt=datefmt)

    #print('XXX: dir(logging):', dir(logging))
    #print('XXX: logging._handlerList:', logging._handlerList)
    #print('XXX: handler:', dir(logging._handlerList[0]))
    #print('XXX: dir(logging.Formatter()):', dir(logging.Formatter()))

#------------------------------

class MyLogFilter(logging.Filter):
    """Can be used to intercept all messages.
    """
    def filter(self, record):
        if not record.args:
            if record.levelno == logging.WARNING:
                print('LogFilter: ', record.name, record.levelname, record.created, record.msg) #record.__dict__)

                formatter = logging.Formatter()
                print('MyLogFilter formatter: ', formatter.format(record))
                print('MyLogFilter formatter: ', formatter.formatMessage(record))
                print('MyLogFilter formatter: ', formatter.formatTime(record))
                print('MyLogFilter formatter: ', formatter.formatStack(record))
                print('MyLogFilter formatter: ', formatter._fmt)
                print('MyLogFilter formatter: ', formatter.datefmt)

        return True

#------------------------------

if __name__ == "__main__" :
  def test_logger(level) :

    #from psana.pyalgos.generic.logger import logging, config_logger

    config_logger(loglevel=level)#, filename='log.txt')

    logger = logging.getLogger(__name__)
    logger.addFilter(MyLogFilter())

    logger.debug    ('Test message logger.debug   ')
    logger.info     ('Test message logger.info    ')
    logger.warning  ('Test message logger.warning ')
    logger.error    ('Test message logger.error   ')
    logger.critical ('Test message logger.critical')
    logger.exception('Test message logger.exception')
    logger.log(logging.DEBUG, 'This is a test message logger.log(logging.DEBUG,msg)')

    #print('XXX: dir(logger):', dir(logger))
    #print('XXX: list of handlers: ', logger.handlers)

#------------------------------

if __name__ == "__main__" :
    import sys
    level = sys.argv[1] if len(sys.argv) > 1 else 'DEBUG'
    test_logger(level)
    sys.exit('End of test')

#------------------------------
