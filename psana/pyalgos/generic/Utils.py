#------------------------------
"""
:py:class:`Utils` - a set of generic utilities
==============================================

Usage::

    # assuming that $PYTHONPATH=.../lcls2/psana
    # Import
    import pyalgos.generic.Utils as gu

    # Methods
    #resp = gu.<method(pars)>

    ts    = gu.str_tstamp(fmt='%Y-%m-%dT%H:%M:%S', time_sec=None)
    usr   = gu.get_enviroment(env='USER')
    usr   = gu.get_login()
    host  = gu.get_hostname()
    cwd   = gu.get_cwd()
    rec   = gu.log_rec_on_start()
    fmode = gu.file_mode(fname)

    gu.create_directory(dir, mode=0o777)
    exists = gu.create_path(path, depth=6, mode=0o777, verb=True)

    arr  = gu.load_textfile(path)
    gu.save_textfile(text, path, mode='w') # mode: 'w'-write, 'a'-append 

See:
    - :py:class:`Utils`
    - :py:class:`NDArrUtils`
    - :py:class:`Graphics`

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created: 2018-01-25 by Mikhail Dubrovin
"""
#--------------------------------

import os
import sys
import getpass
import socket
from time import localtime, strftime, time

import numpy as np

#------------------------------

import logging
log = logging.getLogger('Utils')

#------------------------------

def str_tstamp(fmt='%Y-%m-%dT%H:%M:%S', time_sec=None) :
    """Returns string timestamp for specified format and time in sec or current time by default
    """
    ts = strftime(fmt, localtime(time_sec))
    #log.debug('str_tstamp: %s' % ts)
    return ts

#------------------------------

def get_enviroment(env='USER') :
    """Returns the value of specified by string name environment variable
    """
    return os.environ[env]

#------------------------------

def get_hostname() :
    """Returns login name
    """
    #return os.uname()[1]
    return socket.gethostname()

#------------------------------

def get_cwd() :
    """Returns current working directory
    """
    return os.getcwd()

#------------------------------

def get_login() :
    """Returns login name
    """
    #return os.getlogin()
    return getpass.getuser()

#------------------------------

def file_mode(fname) :
    """Returns file mode, e.g. 0o40377
    """
    from stat import ST_MODE
    return os.stat(fname)[ST_MODE]

#------------------------------

def log_rec_on_start() :
    """Returns (str) record containing timestamp, login, host, cwd, and command line
    """
    return '\n%s user:%s@%s cwd:%s\n  command:%s'%\
           (str_tstamp(fmt='%Y-%m-%dT%H:%M:%S'), get_login(), get_hostname(), get_cwd(), ' '.join(sys.argv))

#------------------------------

def create_directory(dir, mode=0o377) :
    """Creates directory and sets its mode
    """
    if os.path.exists(dir) :
        log.warning('Directory exists: %s' % dir)
    else :
        os.makedirs(dir)
        os.chmod(dir, mode)
        log.warning('Directory created: %s, mode(oct)=%s' % (dir, oct(mode)))

#------------------------------

def create_path(path, depth=6, mode=0o377) : 
    """Creates missing path of specified depth from the beginning
       e.g. for '/reg/g/psdm/logs/calibman/2016/07/log-file-name.txt'
       or '/reg/d/psdm/cxi/cxi11216/calib/Jungfrau::CalibV1/CxiEndstation.0:Jungfrau.0/pedestals/9-end.data'

       Returns True if path to file exists, False othervise
    """
    log.warning('create_path: %s' % path)

    #subdirs = path.strip('/').split('/')
    subdirs = path.split('/')
    cpath = subdirs[0]
    for i,sd in enumerate(subdirs[:-1]) :
        if i>0 : cpath += '/%s'% sd 
        if i<depth : continue
        if cpath=='' : continue
        create_directory_with_mode(cpath, mode, verb)

    return os.path.exists(cpath)

#------------------------------

def save_textfile(text, path, mode='w') :
    """Saves text in file specified by path. mode: 'w'-write, 'a'-append 
    """
    f=open(path, mode)
    f.write(text)
    f.close() 

#------------------------------

def load_textfile(path) :
    """Returns text file as a str object
    """
    f=open(path, 'r')
    recs = f.read() # f.readlines()
    f.close() 
    return recs

#------------------------------

def replace(template, pattern, subst) :
    """If pattern in the template replaces it with subst.
       Returns str object template with replaced patterns. 
    """
    fields = template.split(pattern, 1) 
    if len(fields) > 1 :
        return '%s%s%s' % (fields[0], subst, fields[1])
    else :
        return template

#------------------------------
#----------- TEST -------------
#------------------------------

def test_01() :    
    #log.debug('debug msg')  # will print a message to the console
    #log.warning('Watch out!')  # will print a message to the console
    #log.info('I told you so')  # will not print anything

    print('get_enviroment("PWD") : %s' % get_enviroment(env='PWD'))
    print('get_hostname()        : %s' % get_hostname())
    print('get_cwd()             : %s' % get_cwd())
    print('get_login()           : %s' % get_login())
    print('str_tstamp()          : %s' % str_tstamp())
    print('log_rec_on_start()    :%s' % log_rec_on_start())
    create_directory('./work', mode=0o377)
    print('file_mode("work")     : %s' % oct(file_mode('work')))
    #print(': %s' % )

#------------------------------

if __name__ == "__main__" :
    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',\
                        datefmt='%m-%d-%Y %H:%M:%S',\
                        level=logging.DEBUG)
                        #filename='example.log', filemode='w'
    test_01()
    sys.exit('\nEnd of test')

#------------------------------
