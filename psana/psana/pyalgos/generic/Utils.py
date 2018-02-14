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

    ts    = gu.str_tstamp(fmt='%Y-%m-%dT%H:%M:%S%z', time_sec=None)
    tsec, ts = gu.time_and_stamp(fmt='%Y-%m-%dT%H:%M:%S%z', time_sec=None)
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

    # Save image in file
    # ==================
    gu.save_image_tiff(image, fname='image.tiff', verb=True) # 16-bit tiff
    gu.save_image_file(image, fname='image.png', verb=True) # gif, pdf, eps, png, jpg, jpeg, tiff (8-bit only)

See:
    - :py:class:`Utils`
    - :py:class:`NDArrUtils`
    - :py:class:`Graphics`

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created: 2018-01-25 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-02
"""
#--------------------------------

import os
import sys
import getpass
import socket
from time import localtime, strftime, time, strptime, mktime
import numpy as np

#------------------------------

import logging
logger = logging.getLogger('Utils')

TSFORMAT = '%Y-%m-%dT%H:%M:%S%z'

#------------------------------

def str_tstamp(fmt='%Y-%m-%dT%H:%M:%S%z', time_sec=None) :
    """Returns string timestamp for specified format and time in sec or current time by default
    """
    ts = strftime(fmt, localtime(time_sec))
    #logger.debug('str_tstamp: %s' % ts)
    return ts

#------------------------------

def time_and_stamp(fmt='%Y-%m-%dT%H:%M:%S%z', time_sec=None) :
    tsec = time() if time_sec is None else time_sec
    return tsec, str_tstamp(fmt, tsec)

#------------------------------

def time_sec_from_stamp(fmt='%Y-%m-%dT%H:%M:%S%z', time_stamp='1970-01-01T00:00:00-0800') :
    try : struc = strptime(time_stamp, fmt)
    except ValueError as err: 
        logger.exception(err)
        sys.exit()
    return int(mktime(struc))

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
           (str_tstamp(fmt='%Y-%m-%dT%H:%M:%S%z'), get_login(), get_hostname(), get_cwd(), ' '.join(sys.argv))

#------------------------------

def create_directory(dir, mode=0o377) :
    """Creates directory and sets its mode
    """
    if os.path.exists(dir) :
        logger.warning('Directory exists: %s' % dir)
    else :
        os.makedirs(dir)
        os.chmod(dir, mode)
        logger.warning('Directory created: %s, mode(oct)=%s' % (dir, oct(mode)))

#------------------------------

def create_path(path, depth=6, mode=0o377) : 
    """Creates missing path of specified depth from the beginning
       e.g. for '/reg/g/psdm/logs/calibman/2016/07/log-file-name.txt'
       or '/reg/d/psdm/cxi/cxi11216/calib/Jungfrau::CalibV1/CxiEndstation.0:Jungfrau.0/pedestals/9-end.data'

       Returns True if path to file exists, False othervise
    """
    logger.warning('create_path: %s' % path)

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

def save_textfile(text, path, mode='w', verb=False) :
    """Saves text in file specified by path. mode: 'w'-write, 'a'-append 
    """
    msg = 'save_textfile %s' % path
    if verb : print(msg)
    logger.debug(msg)

    f=open(path, mode)
    f.write(text)
    f.close() 

#------------------------------

def load_textfile(path, verb=False) :
    """Returns text file as a str object
    """
    msg = 'load_textfile %s' % path
    if verb : print(msg)
    logger.debug(msg)

    f=open(path, 'r')
    recs = f.read() # f.readlines()
    f.close() 
    return recs

#------------------------------

def save_image_tiff(image, fname='image.tiff', verb=False) :
    """Saves image in 16-bit tiff file
    """
    import Image
    msg = 'save_image_tiff %s' % fname
    if verb : print(msg)
    logger.debug(msg)

    img = Image.fromarray(image.astype(np.int16))
    img.save(fname)

#------------------------------

def save_image_file(image, fname='image.png', verb=False) :
    """Saves files with type by extension gif, pdf, eps, png, jpg, jpeg, tiff (8-bit only),
       or txt for any other type
    """
    import scipy.misc as scim

    msg = 'save_image_file %s' % fname
    fields = os.path.splitext(fname)
    if len(fields)>1 and fields[1] in ['.gif', '.pdf', '.eps', '.png', '.jpg', '.jpeg', '.tiff'] : 
        scim.imsave(fname, image) 
    else :
        fnametxt = '%s.txt' % fname
        msg = 'save_image_file: non-supported file extension. Save image in text file %s' % fnametxt
        np.savetxt(fnametxt, image, fmt='%8.1f', delimiter=' ', newline='\n')
        #raise IOError('Unknown file type in extension %s' % fname)
    if verb : print(msg)
    logger.debug(msg)

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

def print_command_line_parameters(parser) :
    """Prints input arguments and optional parameters"""
    (popts, pargs) = parser.parse_args()
    args = pargs                             # list of positional arguments
    opts = vars(popts)                       # dict of options
    defs = vars(parser.get_default_values()) # dict of default options

    print('Command:\n ', ' '.join(sys.argv)+\
          '\nArgument list: %s\nOptional parameters:\n' % str(args)+\
          '  <key>      <value>              <default>')
    for k,v in opts.items() :
        print('  %s %s %s' % (k.ljust(10), str(v).ljust(20), str(defs[k]).ljust(20)))

#------------------------------
#----------- TEST -------------
#------------------------------

def test_10() :
    from psana.pyalgos.generic.NDArrGenerators import random_standard

    image = random_standard()
    verbosity=True
    save_image_tiff(image, fname='image.tiff', verb=verbosity)
    save_image_file(image, fname='image.png',  verb=verbosity)
    save_image_file(image, fname='image.xyz',  verb=verbosity)

#------------------------------

def test_01() :    
    #logger.debug('debug msg')  # will print a message to the console
    #logger.warning('Watch out!')  # will print a message to the console
    #logger.info('I told you so')  # will not print anything

    print('get_enviroment("PWD") : %s' % get_enviroment(env='PWD'))
    print('get_hostname()        : %s' % get_hostname())
    print('get_cwd()             : %s' % get_cwd())
    print('get_login()           : %s' % get_login())
    print('str_tstamp()          : %s' % str_tstamp(fmt='%Y-%m-%dT%H:%M'))
    print('str_tstamp()          : %s' % str_tstamp(fmt='%Y-%m-%dT%H:%M:%S%z'))
    create_directory('./work', mode=0o377)
    print('file_mode("work")     : %s' % oct(file_mode('work')))
    print('log_rec_on_start()    :%s' % log_rec_on_start())

#------------------------------

if __name__ == "__main__" :
    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',\
                        datefmt='%Y-%m-%dT%H:%M:S',\
                        level=logging.DEBUG)
                        #filename='example.log', filemode='w'
    test_01()
    sys.exit('\nEnd of test')

#------------------------------
