"""
:py:class:`UtilsEpix` - utilities for epix... detectors
=======================================================

Usage::
    from psana.detector.UtilsEpix import create_directory, alias_for_id,\
         id_for_alias, id_epix, print_object_dir, 

This software was developed for the LCLS project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2018-02-22 by Mikhail Dubrovin
2020-12-04 - begin conversion to LCLS2
"""

#import os
#from time import time

import logging
logger = logging.getLogger(__name__)

import os

#from psana.pyalgos.generic.NDArrUtils import info_ndarr # print_ndarr
#from psana.pyalgos.generic.Utils import load_textfile, save_textfile, file_mode#, create_path, create_directory
#from psana.pyalgos.generic.GlobalUtils import load_textfile, save_textfile, file_mode#, create_path, create_directory
from psana.pyalgos.generic.Utils import save_textfile, load_textfile #log_rec_on_start, str_tstamp, create_directory, save_textfile, set_file_access_mode

# default parameters
#CALIB_REPO_EPIX10KA = '/reg/g/psdm/detector/gains2/epix10ka/panels' #'./panels'
CALIB_REPO_EPIX10KA = '/cds/group/psdm/detector/gains2/epix10ka/panels'
FNAME_PANEL_ID_ALIASES = '.aliases.txt'

#----

def alias_for_id(panel_id, fname=FNAME_PANEL_ID_ALIASES):
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
    rec = '%04d %s\n' % (alias_max+1, panel_id)
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

#----
