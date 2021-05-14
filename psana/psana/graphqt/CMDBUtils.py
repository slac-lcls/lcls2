
"""Class :py:class:`CMDBBUtils` utilities for calib manager DB methods
==============================================================================

Usage ::
    # Test: python lcls2/psana/psana/graphqt/CMDBUtils.py

    # Import
    from psana.graphqt.CMDBUtils import dbu
See:
  - :class:`CMWMain`
  - `on github <https://github.com/slac-lcls/lcls2>`_.

Created on 2021-05-14 by Mikhail Dubrovin
use parameters --webint to switch code between CMDBUtilsMongo.py and CMDBUtilsWeb.py
"""

import logging
logger = logging.getLogger(__name__)
#from psana.pyalgos.generic.Logger import logger
#import psana.graphqt.CMDBUtils as dbu

from psana.graphqt.CMConfigParameters import cp

logger.debug('in webint %s' % cp.kwargs.get('webint', True))

global dbu
if cp.kwargs.get('webint', True): import psana.graphqt.CMDBUtilsWeb   as dbu
else:                             import psana.graphqt.CMDBUtilsMongo as dbu
