
"""
:py:class:`UtilsEpix` - utilities for epix... detectors
=======================================================

Usage::
    from psana2.detector.UtilsEpix import create_directory, alias_for_id,\
         id_for_alias, id_epix, print_object_dir,

This software was developed for the LCLS project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2018-02-22 by Mikhail Dubrovin
2020-12-04 - begin conversion to LCLS2
"""

import logging
logger = logging.getLogger(__name__)

import os
from psana2.detector.Utils import save_textfile, load_textfile, get_login, str_tstamp
from psana2.detector.dir_root import DIR_REPO_EPIX10KA  # DIR_ROOT, DIR_LOG_AT_START, HOSTNAME
#DIR_REPO_EPIX10KA = DIR_ROOT + '/detector/gains2/epix10ka/panels'
FNAME_PANEL_ID_ALIASES = '.aliases.txt'

def alias_for_id(panel_id, fname=FNAME_PANEL_ID_ALIASES, exp=None, run=None, **kwa):
    """Returns Epix100a/10ka panel short alias for long panel_id,
       e.g., for panel_id = 3925999616-0996663297-3791650826-1232098304-0953206283-2655595777-0520093719
       returns 0001
    """
    alias_max = 0
    if os.path.exists(fname):
      #logger.debug('search alias for panel id: %s\n  in file %s' % (panel_id, fname))
      recs = load_textfile(fname).strip('\n').split('\n')
      for r in recs:
        if not r: continue # skip empty records
        fields = r.strip('\n').split(' ')
        if fields[1] == panel_id:
            logger.debug('found alias %s for panel_id %s\n  in file %s' % (fields[0], panel_id, fname))
            return fields[0]
        ialias = int(fields[0])
        if ialias>alias_max: alias_max = ialias
        #print(fields)
    # if record for panel_id is not found yet, add it to the file and return its alias
    rec = '%04d %s %s' % (alias_max+1, panel_id, str_tstamp())
    if exp is not None: rec += ' %10s' %  exp
    if run is not None: rec += ' r%04d' %  run
    if len(kwa)>0: rec += ' '+' '.join([str(v) for k,v in kwa.items() if v is not None])
    rec += ' %s\n' % get_login()
    logger.debug('file "%s" is appended with record:\n%s' % (fname, rec))
    save_textfile(rec, fname, mode='a')
    return '%04d' % (alias_max+1)


def id_for_alias(alias, fname=FNAME_PANEL_ID_ALIASES):
    """Returns Epix100a/10ka panel panel_id for specified alias,
       e.g., for alias = 0001
       returns 3925999616-0996663297-3791650826-1232098304-0953206283-2655595777-0520093719
    """
    logger.debug('search panel id for alias: %s\n  in file %s' % (alias, fname))
    recs = load_textfile(fname).strip('\n').split('\n')
    for r in recs:
        fields = r.strip('\n').split(' ')
        if fields[0] == alias:
            logger.debug('found panel id %s' % (fields[1]))
            return fields[1]

# EOF
