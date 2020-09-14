#------------------------------

import logging
logger = logging.getLogger(__name__)

from psana.pyalgos.generic.Utils import os_system, os_command

#------------------------------

def command_add(fname, ctype, **kwa):
    """returns (str) command to add consatants from file to the DB,
       ex.: cdb add -e testexper -d testdet_1234 -c test_ctype -r 123 -f cm-confpars.txt -i txt -l DEBUG
    """
    exp       = kwa.get('experiment', None)
    det       = kwa.get('detname',    None)
    runnum    = kwa.get('runnum',     None)
    timestamp = kwa.get('timestamp',  None)
    time_sec  = kwa.get('time_sec',   None)
    version   = kwa.get('version',    None)
    dtype     = kwa.get('dtype',      None)
    comment   = kwa.get('coment',     None)
    loglev    = kwa.get('loglev',     None)
    confirm   = kwa.get('cdbadd',     True)

    cmd = 'cdb add'
    if exp       is not None: cmd += ' -e %s' % exp
    if det       is not None: cmd += ' -d %s' % det
    if ctype     is not None: cmd += ' -c %s' % ctype.ljust(12)
    if dtype     is not None: cmd += ' -i %s' % dtype
    if runnum    is not None: cmd += ' -r %s' % str(runnum)
    if timestamp is not None: cmd += ' -t %s' % timestamp
    if fname     is not None: cmd += ' -f %s' % fname
    if loglev    is not None: cmd += ' -l %s' % loglev
    if version   is not None: cmd += ' -v %s' % version
    if comment   is not None: cmd += ' -m %s' % comment
    if time_sec  is not None: cmd += ' -s %s' % str(time_sec)
    if confirm:               cmd += ' -C'

    logger.debug('command: %s' % cmd)
    return cmd

#------------------------------

def add_constants(fname, ctype, **kwa):
    """execute os command to add consatants from file to the DB,
       ex.: cdb add -e testexper -d testdet_1234 -c test_ctype -r 123 -f cm-confpars.txt -i txt -l DEBUG
    """
    cmd = command_add(fname, ctype, **kwa)
    os_command(cmd) 
    logger.info('executed command: %s' % cmd)

#------------------------------

