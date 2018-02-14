#------------------------------
"""
:py:class:`Logger` - logbook message storage
==============================================

Usage ::

    from psana.pyalgos.generic.Logger import logger

    logger.setLevel('warning') # level: 'debug','info','warning','error','critical'

    pbits = 0o177777 # print control bits: 1,2,4,8,16 for 'debug','info','warning','error','critical'
    logger.setPrintBits(pbits)

    levels = logger.getListOfLevels()
    level  = logger.getLevel()
    fname  = logger.getLogFileName()
    fnamet = logger.getLogTotalFileName()
    tss    = logger.getStrStartTime()
    ts     = logger.timeStamp()
    log    = logger.getLogContent()
    logtot = logger.getLogContentTotal()

    logger.debug   ('This is a test message 1', __name__)
    logger.info    ('This is a test message 2', __name__)
    logger.warning ('This is a test message 3', __name__)
    logger.error   ('This is a test message 4', __name__)
    logger.critical('This is a test message 5', __name__)

    logger.saveLogInFile(fname=None)
    logger.saveLogTotalInFile(fname=None)

    logger.setGUILogger(gui) # will callback guilogger.appendGUILog(msg)

See:
    - :py:class:`Logger`
    - :py:class:`ConfigParameters`
    - `matplotlib <https://matplotlib.org/contents.html>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Adopted for LCLS2 on 2018-02-09 by Mikhail Dubrovin
"""
#------------------------------

import os
from time import localtime, strftime

#------------------------------

class Logger :
    """Logbook for messages.
    """
    levels = ['debug','info','warning','error','critical']

    def __init__ ( self, fname=None, level='info', print_bits=0 ) :
        """Constructor.
           - fname - string file name for output log file
           - level - string level from the list of levels which is used as a threshold for accumulated messages
           - print_bits - terminal output bits: 1,2,4,8,16 for 'debug','info','warning','error','critical'
        """
        self.name = self.__class__.__name__
        self.guilogger = None

        self.print_bits = print_bits
        self.setLevel(level)
        self.selectionIsOn = True # It is used to get total log content
        
        self.log = []
        self._startLog(fname)


    def setLevel(self, level):
        """Sets the threshold level of messages for record selection algorithm"""
        self.level_thr_str = level
        self.level_thr_ind = self.levels.index(level)


    def setPrintBits(self, print_bits):
        """Sets terminal printout bits: 1,2,4,8,16 for 'debug','info','warning','error','critical' 
        """
        self.print_bits = print_bits


    def getListOfLevels(self):
        return self.levels


    def getLevel(self):
        return self.level_thr_str


    def getLogFileName(self):
        return self.fname


    def getLogTotalFileName(self):
        return self.fname_total


    def getStrStartTime(self):
        return self.str_start_time


    def debug   (self, msg, name=None) : self._message(msg, 0, name)

    def info    (self, msg, name=None) : self._message(msg, 1, name)

    def warning (self, msg, name=None) : self._message(msg, 2, name)

    def error   (self, msg, name=None) : self._message(msg, 3, name)

    def critical(self, msg, name=None) : self._message(msg, 4, name)

    def _message(self, msg, index, name=None) :
        """Store input message the 2D tuple of records, send request to append GUI.
        """
        tstamp    = self.timeStamp()
        level     = self.levels[index] 
        rec       = [tstamp, level, index, name, msg]
        self.log.append(rec)

        if self._recordIsSelected(rec) :         
            str_msg = self._stringForRecord(rec)
            self._appendGUILog(str_msg)
            #print(str_msg)

        if self.print_bits & (1<<index) :
            print(self._stringForRecord(rec))


    def _recordIsSelected(self, rec):
        """Apply selection algorithms for each record:
           returns True if the record is passed,
                   False - the record is discarded from selected log content.
        """
        if not self.selectionIsOn       : return True
        if rec[2] < self.level_thr_ind  : return False
        else                            : return True


    def _stringForRecord(self, rec):
        """Returns the strind presentation of the log record, which intrinsically is a tuple."""
        tstamp, level, index, name, msg = rec
        if name is not None :
            return '%s (%s) %s: %s' % (tstamp, level, name, msg)
        else :
            return '%s' % msg


    def _appendGUILog(self, msg='') :
        """Append message in GUI, if it is available"""
        if self.guilogger is None : return

        try    : self.guilogger.appendGUILog(msg)
        except : pass


    def setGUILogger(self, guilogger) :
        """Receives the reference to GUI"""
        self.guilogger = guilogger


    def timeStamp(self, fmt='%Y-%m-%d %H:%M:%S') : # '%Y-%m-%d %H:%M:%S %Z'
        return strftime(fmt, localtime())


    def _startLog(self, fname=None) :
        """Logger initialization at start"""
        self.str_start_time = self.timeStamp( fmt='%Y-%m-%d-%H:%M:%S' )
        if  (not fname) or (fname is None) :
            self.fname       = '%s-log.txt'       % self.str_start_time
            self.fname_total = '%s-log-total.txt' % self.str_start_time
        else :
            self.fname       = fname
            self.fname_total = self.fname + '-total' 

        self.info ('Start session log file: ' + self.fname,       self.name)
        self.debug('Total log file name: '    + self.fname_total, self.name)


    def getLogContent(self):
        """Return the text content of the selected log records"""
        self.log_txt = ''
        for rec in self.log :
            if self._recordIsSelected(rec) :         
                self.log_txt += self._stringForRecord(rec) + '\n'
        return  self.log_txt


    def getLogContentTotal(self):
        """Return the text content of all log records"""
        self.selectionIsOn = False
        log_txt = self.getLogContent()
        self.selectionIsOn = True
        return log_txt


    def saveLogInFile(self, fname=None, mode=0o666):
        """Save content of the selected log records in the text file"""
        if fname is None : fname_log = self.fname
        else             : fname_log = fname
        self._saveTextInFile(self.getLogContent(), fname_log, mode)


    def saveLogTotalInFile(self, fname=None, mode=0o666):
        """Save content of all log records in the text file"""
        if fname is None : fname_log = self.fname_total
        else             : fname_log = fname
        self._saveTextInFile(self.getLogContentTotal(), fname_log, mode)


    def _saveTextInFile(self, text, fname='log.txt', mode=0o666):
        self.debug('saveTextInFile: ' + fname, self.name)
        f=open(fname,'w')
        f.write(text)
        f.close()
        os.chmod(fname, mode)

#-----------------------------

logger = Logger(fname=None)

#-----------------------------

def test_Logger() :

    #logger.setLevel('debug')
    logger.setLevel('warning')
    logger.setPrintBits(0o177777) # print messages
    
    logger.debug   ('This is a test message 1', __name__)
    logger.info    ('This is a test message 2', __name__)
    logger.warning ('This is a test message 3', __name__)
    logger.error   ('This is a test message 4', __name__)
    logger.critical('This is a test message 5', __name__)
    logger.critical('This is a test message 6')

    #logger.saveLogInFile()
    #logger.saveLogTotalInFile()

    print('getLogContent():\n',      logger.getLogContent())
    print('getLogContentTotal():\n', logger.getLogContentTotal())

#-----------------------------

if __name__ == "__main__" :
    import sys
    test_Logger()
    sys.exit('End of test')

#-----------------------------
