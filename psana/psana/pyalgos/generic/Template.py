#------------------------------
"""
:py:class:`Template` - 
==============================================

Usage::

    # Import
    import psana.pyalgos.generic.Template as xx

    # Methods
    #resp = xx.<method(pars)>

See:
    - :py:class:`Utils`
    - :py:class:`NDArrUtils`
    - :py:class:`Graphics`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2018-02-09 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-09
"""
#--------------------------------

import os
import sys
import numpy as np
import psana.pyalgos.generic.Template as gtmp

#------------------------------

import logging
logger = logging.getLogger('Template')

#------------------------------

def test_01():
    pass

#------------------------------

if __name__ == "__main__" :
    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',\
                        datefmt='%Y-%m-%dT%H:%M:S', level=logging.DEBUG) #filename='example.log', filemode='w'
    test_01()
    sys.exit('\nEnd of test')

#------------------------------
