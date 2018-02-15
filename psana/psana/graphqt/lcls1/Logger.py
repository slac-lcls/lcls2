#------------------------------
"""Logger - logger for graphqt

Usage::

    from CalibManager.Logger import logger as log
    # or
    from graphqt.Logger import log

    # set level: 'debug','info','warning','error','critical'
    log.setLevel('warning') 

    # print messages of all levels: 1,2,4,8,16 for 'debug','info',...
    log.setPrintBits(0377) 

    log.debug   ('Some message', __name__)
    log.info    ('Some message', __name__)
    log.warning ('Some message', __name__)
    log.error   ('Some message', __name__)
    log.critical('Some message', __name__)

    levels = logger.getListOfLevels()
    level  = logger.getLevel()
    fname  = logger.getLogFileName()
    fnamet = logger.getLogTotalFileName()
    tss    = logger.getStrStartTime()
    ts     = logger.timeStamp()
    log    = logger.getLogContent()
    logtot = logger.getLogContentTotal()

    log.saveLogInFile()
    log.saveLogTotalInFile()
    # or
    log.saveLogInFile('file.txt')

    log.setGUILogger(guilogger) # will callback guilogger.appendGUILog(msg)


@see class :py:class:`CalibManager.Logger`

@see project modules
    * :py:class:`CalibManager.GUILogger.py`
    * :py:class:`CalibManager.ConfigParameters`

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

@version $Id:Logger.py 11923 2016-05-17 21:14:33Z dubrovin@SLAC.STANFORD.EDU $

@author Mikhail S. Dubrovin
"""
#------------------------------

from CalibManager.Logger import logger as log

#------------------------------

def test_log() :

    print '__name__:', __name__

    # set level: 'debug','info','warning','error','critical'
    log.setLevel('warning') 

    # print messages of all levels: 1,2,4,8,16 for 'debug','info',...
    log.setPrintBits(0377) 
    
    log.debug   ('This is a test message 1', __name__)
    log.info    ('This is a test message 2', __name__)
    log.warning ('This is a test message 3', __name__)
    log.error   ('This is a test message 4', __name__)
    log.critical('This is a test message 5', __name__)
    log.critical('This is a test message 6')
    log.info    ('This is a test message 7','test_log')

    print 'getLogContent():\n',      log.getLogContent()
    print 'getLogContentTotal():\n', log.getLogContentTotal()

    #log.saveLogInFile()
    #log.saveLogTotalInFile()

#------------------------------

if __name__ == "__main__" :
    test_log()

#------------------------------
