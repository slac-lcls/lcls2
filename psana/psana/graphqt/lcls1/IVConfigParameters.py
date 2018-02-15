#!@PYTHON@
#------------------------------
"""
Class :py:class:`IVConfigParameters` supports configuration parameters for application
======================================================================================

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Usage ::

    from expmon.Logger import log

    log.setPrintBits(0377)
    cp.readParametersFromFile()
    cp.printParameters()
    cp.log_level.setValue('debug')
    cp.saveParametersInFile()

See:
    - :class:`IVMain`
    - :class:`IVMainTabs`
    - :class:`IVMainButtons`
    - :class:`IVImageCursorInfo`
    - :class:`IVConfigParameters`
    - :class:`IVTabDataControl`
    - :class:`IVTabFileName`
    - `graphqt documentation <https://lcls-psana.github.io/graphqt/py-modindex.html>`_.

Created on 2016-11-22 by Mikhail Dubrovin
"""
#------------------------------

# import os
#from graphqt.Logger import log
from expmon.PSConfigParameters import PSConfigParameters
from expmon.PSNameManager import nm # It is here for initialization

#------------------------------

class IVConfigParameters(PSConfigParameters) :
    """A storage of configuration parameters for Image Vievier (iv)
    """
    _name = 'IVConfigParameters'
 
    def __init__(self, fname=None) :
        """fname : str - the file name with configuration parameters, if not specified then use default.
        """
        #log.setPrintBits(0377)
        #log.debug('In c-tor', self._name)
        print 'In %s c-tor' % self._name

        PSConfigParameters.__init__(self)

        #self.fname_cp = '%s/%s' % (os.path.expanduser('~'), '.confpars-montool.txt') # Default config file name
        self.fname_cp = './iv-confpars.txt' # Default config file name

        self.declareParameters()
        self.readParametersFromFile()
        #self.printParameters()

        self.ivmain = None

        self.list_of_sources = None # if None - updated in the ThreadWorker

        nm.set_config_pars(self)

#------------------------------
        
    def declareParameters(self) :
        # Possible typs for declaration : 'str', 'int', 'long', 'float', 'bool'
        self.log_level = self.declareParameter(name='LOG_LEVEL_OF_MSGS', val_def='info', type='str')
        #self.log_file  = self.declareParameter(name='LOG_FILE_NAME', val_def='/reg/g/psdm/logs/montool/log.txt', type='str')
        self.log_file  = self.declareParameter(name='LOG_FILE_NAME', val_def='iv-log.txt', type='str')

        self.save_log_at_exit = self.declareParameter( name='SAVE_LOG_AT_EXIT', val_def=True,  type='bool')
        #self.dir_log_cpo      = self.declareParameter( name='DIR_FOR_LOG_FILE_CPO', val_def='/reg/g/psdm/logs/calibman', type='str')

        self.main_win_pos_x  = self.declareParameter(name='MAIN_WIN_POS_X',  val_def=5,    type='int')
        self.main_win_pos_y  = self.declareParameter(name='MAIN_WIN_POS_Y',  val_def=5,    type='int')
        self.main_win_width  = self.declareParameter(name='MAIN_WIN_WIDTH',  val_def=1200, type='int')
        self.main_win_height = self.declareParameter(name='MAIN_WIN_HEIGHT', val_def=700,  type='int')

        self.color_table_ind = self.declareParameter(name='COLOR_TABLE_IND', val_def=1, type='int')
        self.current_tab     = self.declareParameter(name='MAIN_CURRENT_TAB', val_def='Status', type='str')
        self.fname_img       = self.declareParameter(name='FNAME_IMAGE', val_def='',     type='str') # '/reg/d/

#------------------------------

cp = IVConfigParameters()

#------------------------------

def test_IVConfigParameters() :
    from expmon.Logger import log

    log.setPrintBits(0377)
    cp.readParametersFromFile()
    cp.printParameters()
    cp.log_level.setValue('debug')
    cp.saveParametersInFile()

#------------------------------

if __name__ == "__main__" :
    import sys
    test_IVConfigParameters()
    sys.exit(0)

#------------------------------
