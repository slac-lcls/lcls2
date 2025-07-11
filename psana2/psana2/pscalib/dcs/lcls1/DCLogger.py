####!/usr/bin/env python
#------------------------------
"""
Class :py:class:`DCLogger` - logger for Detector Calibration Store
==================================================================

See:
    * :py:class:`DCStore`
    * :py:class:`DCType`
    * :py:class:`DCRange`
    * :py:class:`DCVersion`
    * :py:class:`DCBase`
    * :py:class:`DCInterface`
    * :py:class:`DCUtils`
    * :py:class:`DCDetectorId`
    * :py:class:`DCConfigParameters`
    * :py:class:`DCFileName`
    * :py:class:`DCLogger`
    * :py:class:`DCMethods`
    * :py:class:`DCEmail`

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Created: 2016-05-17 by Mikhail Dubrovin
"""
#------------------------------

from CalibManager.Logger import logger as log

#------------------------------

def test_log() :

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

    print 'getLogContent():\n',      log.getLogContent()
    print 'getLogContentTotal():\n', log.getLogContentTotal()

    #log.saveLogInFile()
    #log.saveLogTotalInFile()

#------------------------------

if __name__ == "__main__" :
    test_log()

#------------------------------
