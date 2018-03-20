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

LOGLEVELS = {'info'    : logging.INFO, 
             'debug'   : logging.DEBUG,
             'warning' : logging.WARNING,
             'error'   : logging.ERROR,
             'critical': logging.CRITICAL}

#------------------------------

def config_logger(loglevel='info',\
                  fmt='%(asctime)s %(name)s %(levelname)s: %(message)s',\
                  datefmt='%Y-%m-%dT%H:%M:%S',\
                  filename='',\
                  filemode='w') :

    level = LOGLEVELS.get(loglevel.lower(), logging.INFO)

    logging.basicConfig(format=fmt, datefmt=datefmt,\
                        level=level, filename=filename, filemode=filemode)

#------------------------------

class MyLogFilter(logging.Filter):
    """Can be used to intercept all messages.
    """
    def filter(self, record):
        if not record.args:
            if record.levelno == logging.WARNING:
                print('LogFilter: ', record.name, record.levelname, record.created, record.msg) #record.__dict__)
        return True

#------------------------------

if __name__ == "__main__" :
  def test_logger(level) :

    #from psana.pyalgos.generic.logger import logging, config_logger

    config_logger(loglevel=level)#, filename='log.txt')

    logger = logging.getLogger('My_Module')
    logger.addFilter(MyLogFilter())

    logger.debug    ('This is a test message 1')
    logger.info     ('This is a test message 2')
    logger.warning  ('This is a test message 3')
    logger.error    ('This is a test message 4')
    logger.critical ('This is a test message 5')
    logger.exception('This is a test message logger.exception')
    logger.log(logging.DEBUG, 'This is a test message logger.log')

#------------------------------

if __name__ == "__main__" :
    import sys
    level = sys.argv[1] if len(sys.argv) > 1 else 'debug'
    test_logger(level)
    sys.exit('End of test')

#------------------------------
