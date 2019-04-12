#------------------------------
"""
:py:class:`NotificationLog` is intended to submit notification records in the log file
======================================================================================

Usage::

    # Import
    from  psana.pyalgos.generic.NotificationLog import NotificationLog

    nl = NotificationLog(fname='test-notification-log.txt', dict_add_fields={'mycom':'my-comment-is-here'})
    nl.add_record(mode='enabled')

See:
    - :py:class:`NotificationLog`
    - :py:class:`Utils`
    - :py:class:`NDArrUtils`
    - :py:class:`Graphics`
    - `matplotlib <https://matplotlib.org/contents.html>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2017-01-27 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-12
"""
__version__ = 'v2018-02-12'

#------------------------------

import sys
import os
#import pwd
import getpass
from time import time, localtime, strftime

#------------------------------

def create_directory(dir, mode=0o777) :
    #print('XXX: create_directory: %s' % dir)
    if os.path.exists(dir) :
        pass
        #print('XXX: Directory exists: ', dir)
        #logger.info('Directory exists: ', dir, __name__) 
    else :
        os.makedirs(dir)
        os.chmod(dir, mode)
        ###os.system(cmd)
        #print('XXX: Directory created: ', dir)
        #logger.info('Directory created: ', dir, __name__) 

#------------------------------

def create_path(path, depth=5, mode=0o777) : 
    # Creates missing path for /reg/g/psdm/logs/calibman/2016/07/2016-07-19-12:20:59-log-dubrovin-562.txt
    # if path to file exists return True, othervise False
    subdirs = path.strip('/').split('/')
    #print('XXX subdirs', subdirs)
    cpath = ''
    for i,sd in enumerate(subdirs[:-1]) :
        cpath += '/%s' % sd 
        #print('create_path %d: %s' % (i, cpath))
        if i < depth : continue
        create_directory(cpath, mode)

    return os.path.exists(cpath)
    
#------------------------------

def note_fname() :
    # Returns name like /reg/g/psdm/logs/logbookgrabber/2016/07/2016-07-19-12:20:59-log-dubrovin-562.txt
    fnote = 'log-notification.txt'
    fname = '2017-01-27-10:00:00-log.txt'  # logger.getLogFileName() # 2016-07-19-11:53:02-log.txt
    year, month = fname.split('-')[:2]  # 2016, 07
    return '%s/%s/%s' % (cp.dir_log.value(), year, fnote)

#-------------------------------

def save_textfile(text, path, mode='w') :
    """Saves text in file specified by path. mode: 'w'-write, 'a'-append 
    """
    f = open(path,mode)
    f.write(text)
    f.close() 

#-------------------------------

class NotificationLog :
    """Is intended to submit notification records in the log file
    """

    def __init__(self, fname='test-notification-log.txt', dict_add_fields={}) :
        self.fname = '%s/%s' % (os.getcwd(), os.path.basename(fname))
        self.dict_add_fields = dict_add_fields        


    def get_info_dict(self) :
        info_dict = {}
        date,time,zone = self.get_current_local_time_stamp().split()
        info_dict['date'] = date
        info_dict['time'] = time
        info_dict['zone'] = zone
        info_dict['user'] = getpass.getuser()
        info_dict['host'] = self.get_enviroment('HOSTNAME') # socket.gethostname()
        info_dict['cwd']  = os.getcwd()
        info_dict['vers'] = self.version()
        info_dict['proc'] = sys.argv[0]
        info_dict['pid']  = '%d' % os.getpid()

        # add user-defined fields
        for k,v in self.dict_add_fields.iteritems() :
            info_dict[k]  = v[1]
        return info_dict


    def get_current_local_time_stamp(self, fmt='%Y-%m-%d %H:%M:%S %Z'):
        return strftime(fmt, localtime())


    def get_enviroment(self, env='USER') :
        return str(os.environ.get(env))
        #return os.environ[env] if env in os.environ.keys() else '%s-NONDEF-ENV' % env


    def version(self) :
        return __version__


    def cname(self) :
        return __author__.split()[2].lower()


    def is_permitted(self) :
        s = self.get_enviroment(env='LOGNAME') == self.cname()
        print('is_permitted:', s)
        return s


    def add_record(self, mode='enabled') : #  mode='self-disabled'
        if mode == 'self-disabled' and self.is_permitted() : return
        d = self.get_info_dict()
        rec = ' '.join(['%s:%s'%(k,str(v)) for k,v in d.iteritems()])
        #print('add_record rec:', rec)
        path = self.fname # note_fname() if self.fname is None else fname
        #print('Note file: %s' % path)
        if create_path(path, depth=5) :
            msg = 'Save record:\n  %s\n  in file: %s' % (rec, path)
            print(msg)
            #logger.info(msg, self.__class__.__name__)
            save_textfile('%s\n'%rec, path, mode='a')
        
        else :
            msg = 'Can not create path to the file: %s' % path
            logger.warning(msg, self.__class__.__name__)
        
#------------------------------
#------------------------------
#----------- TESTS ------------
#------------------------------
#------------------------------
 
def test_notificationlog(tname) :
    _name = sys._getframe().f_code.co_name
    print('In %s' % _name)
    nl = NotificationLog(fname='test-notification-log.txt', dict_add_fields={'mycom':'my-comment-is-here'})
    nl.add_record(mode='enabled')

#------------------------------

if __name__ == "__main__" :
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s' % tname)
    if   tname == '0': test_notificationlog(tname)
    elif tname == '1': test_notificationlog(tname)
    else : sys.exit('Test %s is not implemented' % tname)
    sys.exit ('End of %s' % sys.argv[0])

#------------------------------
