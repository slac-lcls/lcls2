#### !/usr/bin/env python
#------------------------------
"""
:py:class:`CalibBase` - abstract class with interface description
=======================================================================

Usage::

    # python lcls2/psana/psana/pscalib/calib/CalibBase.py

    from psana.pscalib.calib.CalibBase import *
    logger = logging.getLogger('MyModuleName')

    cb = CalibBase()
    cb.put(data, **kwargs)

    cb.put(data,
           calibtype=PEDESTALS,
           detector='cspad-0-cxids2-0',
           experiment='abcd12345',
           run=123,
           version='V0-1-2',
           time_sec=1517608049,
           time_nsec=123456789,
           time_stamp='2018-02-02 13:47:29',
           facility='LCLS2',
           host='undefined',
           uid='undefined',
           pwd='undefined',
           comments=['very good constants', 'throw them in trash immediately!']
          )

    data = cb.get(**kwargs)

See:
 * :py:class:`Calib`
 * :py:class:`CalibConstants`
 * :py:class:`CalibBase`

For more detail see `Calibration Store <https://confluence.slac.stanford.edu/display/PCDS/MongoDB+evaluation+for+calibration+store>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2018-02-02 by Mikhail Dubrovin
"""
#------------------------------

import logging
logger = logging.getLogger('CalibBase')

from psana.pscalib.calib.CalibConstants import *

#------------------------------

class CalibBase() :

    def __init__(self, **kwargs) :
        self.dbname = kwargs.get('dbname', 'Undefined')

    def _show_warning(self, methname='default') :
        msg='Interface method %s needs to be re-implemented in the derived class'%methname
        logger.warning(msg)

    def put(self, data, **kwargs) :
        self._show_warning('put(data)')
        
    def get(self, **kwargs) :
        self._show_warning('get()')
        return None

    def delete(self, **kwargs) :
        self._show_warning('delete()')
        return None

#------------------------------

if __name__ == "__main__" :
    #logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',\
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s',\
                        datefmt='%Y-%m-%dT%H:%M:%S',\
                        level=logging.DEBUG) #filename='example.log', filemode='w'
    o = CalibBase()
    o.put(None)
    data = o.get()

#------------------------------
