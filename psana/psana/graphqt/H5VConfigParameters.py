#------------------------------
"""
Class :py:class:`H5VConfigParameters` supports configuration parameters for application
======================================================================================

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Usage ::

    cp.readParametersFromFile()
    cp.printParameters()
    cp.log_level.setValue('DEBUG')
    cp.saveParametersInFile()

See:
    - :class:`H5VMain`
    - `graphqt documentation <https://lcls-psana.github.io/graphqt/py-modindex.html>`_.

Created on 2019-12-02 by Mikhail Dubrovin
"""
#------------------------------
import os

import logging
logger = logging.getLogger(__name__)

#------------------------------

from psana.pyalgos.generic.PSConfigParameters import PSConfigParameters

#------------------------------

class H5VConfigParameters(PSConfigParameters) :
    """A storage of configuration parameters for LCLS-2 Calibration Manager (CM)
    """
    char_expand    = u' \u25BC' # down-head triangle
    char_shrink    = u' \u25B2' # solid up-head triangle
 
    def __init__(self, fname=None) :
        """fname : str - the file name with configuration parameters, if not specified then use default.
        """
        PSConfigParameters.__init__(self)

        logger.debug('In %s c-tor')

        #self.fname_cp = '%s/%s' % (os.path.expanduser('~'), '.h5v-confpars.txt') # Default config file name
        self.fname_cp = './h5v-confpars.txt' # Default config file name

        self.declareParameters()
        self.readParametersFromFile()
        #self.printParameters()

        # Widgets with direct access
        self.h5vmain = None

#------------------------------
        
    def declareParameters(self) :
        # Possible typs for declaration : 'str', 'int', 'long', 'float', 'bool'
        self.log_level = self.declareParameter(name='LOG_LEVEL', val_def='DEBUG', type='str') # val_def='NOTSET'

        #self.log_file - DEPRICATED
        #self.log_file  = self.declareParameter(name='LOG_FILE_NAME', val_def='h5v-log.txt', type='str')
        #self.log_prefix = self.declareParameter(name='LOG_FILE_PREFIX', val_def='/reg/g/psdm/logs/calibman/lcls2', type='str')

        #prefix = '%s/.hdf5explorer-log' % os.path.expanduser('~')
        #self.log_prefix = self.declareParameter(name='LOG_FILE_PREFIX', val_def=prefix, type='str')
        self.log_prefix = self.declareParameter(name='LOG_FILE_PREFIX', val_def='./h5v-log', type='str')
        self.save_log_at_exit = self.declareParameter(name='SAVE_LOG_AT_EXIT', val_def=True, type='bool')

        self.main_win_pos_x  = self.declareParameter(name='MAIN_WIN_POS_X',  val_def=5,    type='int')
        self.main_win_pos_y  = self.declareParameter(name='MAIN_WIN_POS_Y',  val_def=5,    type='int')
        self.main_win_width  = self.declareParameter(name='MAIN_WIN_WIDTH',  val_def=1200, type='int')
        self.main_win_height = self.declareParameter(name='MAIN_WIN_HEIGHT', val_def=700,  type='int')

        self.main_vsplitter  = self.declareParameter(name='MAIN_VSPLITTER', val_def=600, type='int')

#------------------------------
        
#    def __del__(self) :
#        logger.debug('In d-tor: %s')
#        logger.debug('In H5VConfigParameters d-tor')
#        if self.save_cp_at_exit.value() :
#            self.saveParametersInFile()
#        #ConfigParameters.__del__(self)

#------------------------------

cp = H5VConfigParameters()

#import psana.pscalib.calib.CalibConstants as cc
#cp.user = None #cc.USERNAME
#cp.upwd = ''   #cc.USERPW
#print('XXXX H5VConfigParameters set cp.user: %s p: %s' % (cp.user, cp.upwd))

#------------------------------

def test_H5VConfigParameters() :

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    cp.readParametersFromFile()
    cp.printParameters()
    cp.log_level.setValue('DEBUG')
    cp.saveParametersInFile()

#------------------------------

if __name__ == "__main__" :
    import sys
    test_H5VConfigParameters()
    sys.exit(0)

#------------------------------
