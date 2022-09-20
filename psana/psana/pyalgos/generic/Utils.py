
"""
:py:class:`Utils` - a set of generic utilities
==============================================

Usage::

    # assuming that $PYTHONPATH=.../lcls2/psana

    # Run test: python lcls2/psana/psana/pyalgos/generic/Utils.py 1

    # Import
    from psana.pyalgos.generic.Utils import input_single_char
    import psana.pyalgos.generic.Utils as gu

    # Methods
    #resp = gu.<method(pars)>

    ts    = gu.str_tstamp(fmt='%Y-%m-%dT%H:%M:%S%z', time_sec=None)
    tsec, ts = gu.time_and_stamp(fmt='%Y-%m-%dT%H:%M:%S%z', time_sec=None)
    tsec  = gu.time_sec_from_stamp(fmt='%Y-%m-%dT%H:%M:%S%z', time_stamp='1970-01-01T00:00:00-0800')
    usr   = gu.get_enviroment(env='USER')
    usr   = gu.get_login()
    host  = gu.get_hostname()
    cwd   = gu.get_cwd()
    pid   = gu.get_pid()
    stat  = gu.shell_command_is_available(cmd='mongorestore', verb=True)
    rec   = gu.log_rec_on_start()
    fmode = gu.file_mode(fname)

    gu.create_directory(dir, mode=0o777)
    exists = gu.create_path(path, depth=6, mode=0o777)

    flist = gu.get_list_of_files_in_dir(dirname)
    flist = gu.get_list_of_files_in_dir_for_ext(dir, ext='.xtc')
    flist = gu.get_list_of_files_in_dir_for_pattern(dir, pattern='-r0022')
    owner = gu.get_path_owner(path)
    mode  = gu.get_path_mode(path)
    tmpf  = gu.get_tempfile(mode='r+b',suffix='.txt')

    gu.print_parsed_path(path)

    arr  = gu.load_textfile(path)
    gu.save_textfile(text, path, mode='w') # mode: 'w'-write, 'a'-append
    gu.set_file_access_mode(fname, mode=0o777)

    jo = gu.load_json(fname)
    gu.save_json(jo, fname)
    o = gu.load_pickle(fname)
    gu.save_pickle(o, fname)

    # Save image in file
    # ==================
    gu.save_image_tiff(image, fname='image.tiff', verb=True) # 16-bit tiff
    gu.save_image_file(image, fname='image.png', verb=True) # gif, pdf, eps, png, jpg, jpeg, tiff (8-bit only)

    list_int = gu.list_of_int_from_list_of_str(list_str)
    list_str = gu.list_of_str_from_list_of_int(list_int, fmt='%04d')

    resp = gu.has_kerberos_ticket()
    resp = gu.check_token(do_print=False)
    resp = gu.get_afs_token(do_print=False)
    hlst = gu.list_of_hosts_from_lshosts(filter='ps')
    resp = gu.text_sataus_of_lsf_hosts(farm='psnehfarm')
    resp = gu.ext_status_of_queues(lst_of_queues=['psanaq', 'psnehq', 'psfehq', 'psnehprioq', 'psfehprioq'])

    gu.str_kwargs(kwargs, title='Input parameters:', fmt='\n%20s : %s'):
    gu.print_kwargs(kwargs)
    gu.print_parser(parser) # from optparse import OptionParser

    s = gu.do_print(nev) # returns true for sparcified event numbers.
    ch = gu.input_single_char('Next event? [y/n]')

    os_system(cmd)
    os_command(cmd)

See:
    - :py:class:`Utils`
    - :py:class:`PSUtils`
    - :py:class:`NDArrUtils`
    - :py:class:`Graphics`

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created: 2018-01-25 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-02
"""

import os
import sys
import getpass
import socket
from time import localtime, strftime, time, strptime, mktime
import numpy as np
import tty, termios

#import subprocessif

from subprocess import call
if sys.version_info.major == 2:
    from commands import getoutput
else:
    from subprocess import getoutput


# init_logger etc is moved to logger.py
# from psana.pyalgos.generic.logger import init_logger, STR_LEVEL_NAMES, DICT_NAME_TO_LEVEL, TSFORMAT
import logging
logger = logging.getLogger('__name__')


def str_tstamp(fmt='%Y-%m-%dT%H:%M:%S%z', time_sec=None):
    """Returns string timestamp for specified format and time in sec or current time by default
    """
    ts = strftime(fmt, localtime(time_sec))
    #logger.debug('str_tstamp: %s' % ts)
    return ts


def str_tstamp_v1(fmt='%Y-%m-%dT%H:%M:%S.%f%z', time_sec=None):
    """Returns string timestamp for specified format and time in sec or current time by default
    """
    from datetime import datetime
    dt = datetime.fromtimestamp(time() if time_sec is None else time_sec)
    return dt.strftime(fmt)


def time_and_stamp(fmt='%Y-%m-%dT%H:%M:%S%z', time_sec=None):
    tsec = time() if time_sec is None else time_sec
    return tsec, str_tstamp(fmt, tsec)


def time_sec_from_stamp(fmt='%Y-%m-%dT%H:%M:%S%z', time_stamp='1970-01-01T00:00:00-0800'):
    try: struc = strptime(time_stamp, fmt)
    except ValueError as err:
        logger.exception(err)
        sys.exit()
    return int(mktime(struc))


def get_enviroment(env='USER'):
    """Returns the value of specified by string name environment variable
    """
    return os.environ.get(env, None)


def get_hostname():
    """Returns login name
    """
    #return os.uname()[1]
    return socket.gethostname()


def get_cwd():
    """Returns current working directory
    """
    return os.getcwd()


def get_pid():
    """Returns pid - process id
    """
    return os.getpid()


def get_login():
    """Returns login name
    """
    #return os.getlogin()
    return getpass.getuser()


def shell_command_is_available(cmd='mongorestore', verb=True):
    import shutil
    if shutil.which(cmd) is None:
        if verb: logger.warning('shell command "%s" is unavailable.' % cmd)
        return


def file_mode(fname):
    """Returns file mode, e.g. 0o40377
    """
    from stat import ST_MODE
    return os.stat(fname)[ST_MODE]


def log_rec_on_start(tsfmt='%Y-%m-%dT%H:%M:%S%z'):
    """Returns (str) record containing timestamp, login, host, cwd, and command line
    """
    return '\n%s user:%s@%s cwd:%s command:%s'%\
           (str_tstamp(fmt=tsfmt), get_login(), get_hostname(), get_cwd(), ' '.join(sys.argv))


def create_directory(d, mode=0o777, umask=0o0):
    """Creates directory and sets its mode
    """
    os.umask(umask)
    if os.path.exists(d):
        logger.debug('Directory exists: %s' % d)
    else:
        try: os.makedirs(d, mode) #, exist_ok=True)
        except Exception as err:
           logger.debug('exception in os.makedirs("%s") err: %s' % (d,err))
           return
        #os.makedirs(d, mode, exist_ok=True)
        os.chmod(d, mode)
        logger.debug('Directory created: %s, mode(oct)=%s' % (d, oct(mode)))


def create_path(path, depth=6, mode=0o777):
    """Creates missing path of specified depth from the beginning
       e.g. for '/reg/g/psdm/logs/calibman/2016/07/log-file-name.txt'
       or '/reg/d/psdm/cxi/cxi11216/calib/Jungfrau::CalibV1/CxiEndstation.0:Jungfrau.0/pedestals/9-end.data'

       Returns True if path to file exists, False othervise
    """
    logger.debug('create_path: %s' % path)

    #subdirs = path.strip('/').split('/')
    subdirs = path.split('/')
    cpath = subdirs[0]
    for i,sd in enumerate(subdirs[:-1]):
        if i>0: cpath += '/%s'% sd
        if i<depth: continue
        if cpath=='': continue
        create_directory(cpath, mode)

    return os.path.exists(cpath)


def get_list_of_files_in_dir(dirname):
    return os.listdir(dirname)


def get_list_of_files_in_dir_for_ext(dir, ext='.xtc'):
    """Returns the list of files in the directory for specified extension or None if directory is None."""
    if dir is None: return []
    if not os.path.exists(dir): return []

    list_of_files_in_dir = os.listdir(dir)
    list_of_files = []
    for fname in list_of_files_in_dir:
        if os.path.splitext(fname)[1] == ext:
            list_of_files.append(fname)
    return sorted(list_of_files)


def get_list_of_files_in_dir_for_part_fname(dir, pattern='-r0022'):
    """Returns the list of files in the directory for specified file name pattern or [] - empty list."""
    if dir is None: return []
    if not os.path.exists(dir): return []

    list_of_files_in_dir = os.listdir(dir)
    list_of_files = []
    for fname in list_of_files_in_dir:
        if pattern in fname:
            fpath = os.path.join(dir,fname)
            list_of_files.append(fpath)
    return sorted(list_of_files)


def get_path_owner(path):
    import pwd
    stat = os.stat(path)
    #print(' stat =', stat)
    pwuid = pwd.getpwuid(stat.st_uid)
    #print(' pwuid =', pwuid)
    user_name  = pwuid.pw_name
    #print(' uid = %s   user_name  = %s' % (uid, user_name))
    return user_name


def get_path_mode(path):
    return os.stat(path).st_mode


def get_tempfile(mode='r+b',suffix='.txt'):
    import tempfile
    tf = tempfile.NamedTemporaryFile(mode=mode,suffix=suffix)
    return tf # .name


def print_parsed_path(path):                       # Output for path:
    print('print_parsed_path(path): path:',)        # path/reg/d/psdm/XCS/xcsi0112/xtc/e167-r0015-s00-c00.xtc
    print('exists(path)  =', os.path.exists(path))  # True
    print('splitext(path)=', os.path.splitext(path))# ('/reg/d/psdm/XCS/xcsi0112/xtc/e167-r0015-s00-c00', '.xtc')
    print('basename(path)=', os.path.basename(path))# e167-r0015-s00-c00.xtc
    print('dirname(path) =', os.path.dirname(path)) # /reg/d/psdm/XCS/xcsi0112/xtc
    print('lexists(path) =', os.path.lexists(path)) # True
    print('isfile(path)  =', os.path.isfile(path))  # True
    print('isdir(path)   =', os.path.isdir(path))   # False
    print('split(path)   =', os.path.split(path))   # ('/reg/d/psdm/XCS/xcsi0112/xtc', 'e167-r0015-s00-c00.xtc')


def set_file_access_mode(fname, mode=0o777):
    os.chmod(fname, mode)


def save_textfile(text, path, mode='w', verb=False):
    """Saves text in file specified by path. mode: 'w'-write, 'a'-append
    """
    msg = 'save_textfile %s' % path
    if verb: print(msg)
    logger.debug(msg)

    f=open(path, mode)
    f.write(text)
    f.close()


def load_textfile(path, verb=False):
    """Returns text file as a str object
    """
    msg = 'load_textfile %s' % path
    if verb: print(msg)
    logger.debug(msg)

    f=open(path, 'r')
    recs = f.read() # f.readlines()
    f.close()
    return recs


def load_json(fname):
    """Load json object from file.
    """
    logger.debug('load_json %s' % fname)
    import json
    return json.load(open(fname,'rb'))
    # or
    #with open(fname) as f: jo = json.load(f)
    #return jo


def save_json(jo, fname, mode='w'):
    """Saves json object in file.
    """
    logger.debug('save_json %s' % fname)
    import json
    with open(fname, mode) as f: json.dump(jo, f)


def load_pickle(fname, mode='rb'):
    """Returns object from packed in file.
    """
    logger.debug('load_pickle %s' % fname)
    import pickle
    return pickle.load(open(fname, mode))


def save_pickle(o, fname, mode='wb'):
    """Saves object in the pickle file.
    """
    logger.debug('save_pickle %s' % fname)
    import pickle
    with open(fname, mode) as f: pickle.dump(o, f)


def save_image_tiff(image, fname='image.tiff', verb=False):
    """Saves image in 16-bit tiff file
    """
    import Image
    msg = 'save_image_tiff %s' % fname
    if verb: print(msg)
    logger.debug(msg)

    img = Image.fromarray(image.astype(np.int16))
    img.save(fname)


def save_image_file(image, fname='image.png', verb=False):
    """Saves files with type by extension gif, pdf, eps, png, jpg, jpeg, tiff (8-bit only),
       or txt for any other type
    """
    import scipy.misc as scim

    msg = 'save_image_file %s' % fname
    fields = os.path.splitext(fname)
    if len(fields)>1 and fields[1] in ['.gif', '.pdf', '.eps', '.png', '.jpg', '.jpeg', '.tiff']:
        scim.imsave(fname, image)
    else:
        fnametxt = '%s.txt' % fname
        msg = 'save_image_file: non-supported file extension. Save image in text file %s' % fnametxt
        np.savetxt(fnametxt, image, fmt='%8.1f', delimiter=' ', newline='\n')
        #raise IOError('Unknown file type in extension %s' % fname)
    if verb: print(msg)
    logger.debug(msg)


def replace(template, pattern, subst):
    """If pattern in the template replaces it with subst.
       Returns str object template with replaced patterns.
    """
    fields = template.split(pattern, 1)
    if len(fields) > 1:
        return '%s%s%s' % (fields[0], subst, fields[1])
    else:
        return template


def print_command_line_parameters(parser):
    """Prints input arguments and optional parameters"""
    (popts, pargs) = parser.parse_args()
    args = pargs                             # list of positional arguments
    opts = vars(popts)                       # dict of options
    defs = vars(parser.get_default_values()) # dict of default options

    print('Command:\n ', ' '.join(sys.argv)+\
          '\nArgument list: %s\nOptional parameters:\n' % str(args)+\
          '  <key>      <value>              <default>')
    for k,v in opts.items():
        print('  %s %s %s' % (k.ljust(10), str(v).ljust(20), str(defs[k]).ljust(20)))


def list_of_int_from_list_of_str(list_str):
    """Converts  ['0001', '0202', '0203', '0204',...] to [1, 202, 203, 204,...]
    """
    return [int(s) for s in list_str]


def list_of_str_from_list_of_int(list_int, fmt='%04d'):
    """Converts [1, 202, 203, 204,...] to ['0001', '0202', '0203', '0204',...]
    """
    return [fmt % i for i in list_int]


def has_kerberos_ticket():
    """Checks to see if the user has a valid Kerberos ticket"""
    #stream = os.popen('klist -s')
    #output = getoutput('klist -4')
    #resp = call(["klist", "-s"])
    return True if call(["klist", "-s"]) == 0 else False


def _parse_token(token):
    """ from string like: User's (AFS ID 5269) tokens for afs@slac.stanford.edu [Expires Feb 28 19:16] 54 75 Expires Feb 28 19:16
        returns date/time: Feb 28 19:16
    """
    timestamp = ''

    for line in token.split('\n'):
        pos_beg = line.find('[Expire')
        if pos_beg == -1: continue
        pos_end = line.find(']', pos_beg)
        #print(line)
        timestamp = line[pos_beg+9:pos_end]

        #date_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')
        #date_object = datetime.strptime(timestamp, '%b %d %H:%M')
        #print('date_object', str(date_object))

    return timestamp


def check_token(do_print=False):
    token = getoutput('tokens')
    #if do_print(: print(token)
    status = True if 'Expire' in token else False
    timestamp = _parse_token(token) if status else ''
    msg = 'Your AFS token %s %s' % ({True:'IS valid until', False:'IS NOT valid'}[status], timestamp)
    if do_print: print(msg)
    return status, msg


def get_afs_token(do_print=False):
    output = getoutput('aklog')
    if do_print: print(str(output))
    return output


def list_of_hosts(filter='psana'):
    """Returns list of hosts for lshosts"""
    cmd = 'lshosts | grep %s' % filter
    lines = getoutput(cmd).split('\n')
    hosts = [line.split()[0] for line in lines]
    return hosts


def text_sataus_of_lsf_hosts(farm='psnehfarm'):
    """Returns text output of the command: bhosts farm"""
    cmd = 'bhosts %s' % farm
    return cmd, getoutput(cmd)


def text_status_of_queues(lst_of_queues=['psanaq', 'psnehq', 'psfehq', 'psnehprioq', 'psfehprioq']):
    """Checks status of queues"""
    cmd = 'bqueues %s' % (' '.join(lst_of_queues))
    return cmd, getoutput(cmd)


def str_kwargs(kwargs, title='Input parameters:', fmt='\n%20s: %s'):
    return title + ''.join([fmt % (k,str(v)) for k,v in kwargs.items()])


def print_kwargs(kwargs, cmt='%s\n  kwargs:' % (40*'_')):
    print(cmt)
    for k,v in kwargs.items(): print('  %10s: %10s' % (k,v))
    print(40*'_')


def str_attributes(o, cmt='\nattributes:', fmt='\n  %s'):
    return cmt + ''.join([fmt % str(v) for v in dir(o)])


#def str_attributes(o, cmt='\nattributes:', fmt='\n%20s: %s'):
#    return  str(dir(o))
    #return cmt + ''.join([fmt % (k,str(v)) for k,v in dir(o) if len(k)>2 and k[:2] != '__'])


def print_parser(parser):
    """Prints input parameters"""
    popts, pargs = parser.parse_args()
    args = pargs
    opts = vars(popts)
    defs = vars(parser.get_default_values())

    print('Arguments: %s\nOptional parameters:\n' % str(args)+\
          '<key>      <value>          <default>')
    for k,v in opts.items():
        print('%s %s %s' % (k.ljust(10), str(v).ljust(16), str(defs[k]).ljust(16)))


def is_in_command_line(ptrn1=None, ptrn2=None):
    """Returns True (False) if parameter is (not) specified in the command line"""
    if len(sys.argv) < 2: return False
    for p in sys.argv[1:]:
        if ptrn1 is not None and (ptrn1 in p[:2]):
            #logger.debug('option "%s" is found in CL' % ptrn1)
            return True
        if ptrn2 is not None and (ptrn2 in p):
            #logger.debug('option "%s" is found in CL' % ptrn2)
            return True
    return False


def do_print(nev):
    """Returns true for sparcified event numbers.
    """
    return nev<10\
       or (nev<50 and (not nev%10))\
       or (nev<500 and (not nev%100))\
       or not nev%1000


def input_single_char(prompt='input? >'):
    """ input of single character from keybord without <CR>
        import sys, tty, termios
    """
    sys.stdout.write('\r'+prompt)
    sys.stdout.flush()
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setraw(fd)
    ch = sys.stdin.read(1)
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


#def get_grpnames(user='root'):
#    """Returns tuple of group names"""
#    from grp import getgrnam
#    return getgrnam(user)


def os_system(cmd):
    assert isinstance(cmd,str), 'command should be str'
    os.system(cmd)
    logger.debug('os_system command: %s' % cmd)


def os_command(cmd):
    assert isinstance(cmd,str), 'command should be str'
    #_cmd = cmd.split() if isinstance(cmd,str) else cmd
    _cmd = cmd
    stream = os.popen(_cmd)
    resp = stream.read()
    msg = '%s\n%s' % (_cmd, resp) if resp else _cmd
    logger.debug('os_command resp: %s' % msg)


#----------- TEST -------------

if __name__ == "__main__":

  def test_10():
    from psana.pyalgos.generic.NDArrGenerators import random_standard

    image = random_standard()
    verbosity=True
    save_image_tiff(image, fname='image.tiff', verb=verbosity)
    save_image_file(image, fname='image.png',  verb=verbosity)
    save_image_file(image, fname='image.xyz',  verb=verbosity)


  def test_datetime():
    from datetime import datetime
    t_sec = time()
    print('t_sec:', t_sec)
    t = datetime.fromtimestamp(t_sec)
    print('t:', t)
    tnow = datetime.now()
    print('datetime.now:', tnow)
    tstamp = t.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
    zone = strftime('%z', localtime(t_sec))
    print(tstamp)
    print('zone', zone)
    tsz = '%s%s' % (tstamp,zone)
    print('tsz', tsz)


  def test_input_single_char():
    for n in range(20):
      ch = input_single_char('Event:%03d Next event? [y/n]' %n)
      if ch != 'y': sys.exit('\nExit by key %s' % ch)


  def test_01():
    #logger.debug('debug msg')  # will print a message to the console
    #logger.warning('Watch out!')  # will print a message to the console
    #logger.info('I told you so')  # will not print anything

    print('get_enviroment("PWD"): %s' % get_enviroment(env='PWD'))
    print('get_hostname()       : %s' % get_hostname())
    print('get_cwd()            : %s' % get_cwd())
    print('get_login()          : %s' % get_login())
    print('str_tstamp()         : %s' % str_tstamp(fmt='%Y-%m-%dT%H:%M'))
    print('str_tstamp()         : %s' % str_tstamp(fmt='%Y-%m-%dT%H:%M:%S%z'))
    create_directory('./work', mode=0o377)
    print('file_mode("work")    : %s' % oct(file_mode('work')))
    print('log_rec_on_start()   :%s' % log_rec_on_start())
    #print('get_grpnames()       :%s' % str(get_grpnames('root')))
    print('list_of_hosts        :%s' % list_of_hosts())


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',\
                        datefmt='%Y-%m-%dT%H:%M:%S',\
                        level=logging.DEBUG)
                        #filename='example.log', filemode='w'
    test_01()
    test_datetime()
    test_input_single_char()
    sys.exit('\nEnd of test')

# EOF
