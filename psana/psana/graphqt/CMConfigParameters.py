#------------------------------
"""
Class :py:class:`CMConfigParameters` supports configuration parameters for application
======================================================================================

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Usage ::

    from psana.pyalgos.generic.Logger import logger

    logger.setPrintBits(0o377)
    cp.readParametersFromFile()
    cp.printParameters()
    cp.log_level.setValue('debug')
    cp.saveParametersInFile()

See:
    - :class:`CMMain`
    - :class:`CMMainTabs`
    - `graphqt documentation <https://lcls-psana.github.io/graphqt/py-modindex.html>`_.

Created on 2016-11-22 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-26 by Mikhail Dubrovin
"""
#------------------------------

from psana.pyalgos.generic.PSConfigParameters import PSConfigParameters

#from expmon.PSNameManager import nm # It is here for initialization

#------------------------------

class CMConfigParameters(PSConfigParameters) :
    """A storage of configuration parameters for LCLS-2 Calibration Manager (CM)
    """
    def __init__(self, fname=None) :
        """fname : str - the file name with configuration parameters, if not specified then use default.
        """

        PSConfigParameters.__init__(self)

        self._name = 'CMConfigParameters'
        #log.debug('In c-tor', self._name)
        print('In %s c-tor' % self._name)

        #self.fname_cp = '%s/%s' % (os.path.expanduser('~'), '.confpars-montool.txt') # Default config file name
        self.fname_cp = './cm-confpars.txt' # Default config file name

        self.declareParameters()
        self.readParametersFromFile()
        #self.printParameters()

        #nm.set_config_pars(self)

        self.list_of_hosts = ('psanaphi105', 'psanaphi106', 'psanaphi107')
        self.list_of_ports = (27017, 27018, 27019, 27020, 27021)
        self.list_of_str_ports = [str(v) for v in self.list_of_ports]

        # Widgets with direct access
        self.cmwmain     = None
        self.cmwmaintabs = None
        self.cmwdbmain   = None
        self.cmwdbtree   = None

#------------------------------
        
    def declareParameters(self) :
        # Possible typs for declaration : 'str', 'int', 'long', 'float', 'bool'
        self.log_level = self.declareParameter(name='LOG_LEVEL_OF_MSGS', val_def='info', type='str')
        #self.log_file  = self.declareParameter(name='LOG_FILE_NAME', val_def='/reg/g/psdm/logs/montool/log.txt', type='str')
        self.log_file  = self.declareParameter(name='LOG_FILE_NAME', val_def='cm-log.txt', type='str')

        self.save_log_at_exit = self.declareParameter( name='SAVE_LOG_AT_EXIT', val_def=True,  type='bool')
        #self.dir_log_global      = self.declareParameter( name='DIR_LOG_FILE_GLOBAL', val_def='/reg/g/psdm/logs/calibman', type='str')

        self.main_win_pos_x  = self.declareParameter(name='MAIN_WIN_POS_X',  val_def=5,    type='int')
        self.main_win_pos_y  = self.declareParameter(name='MAIN_WIN_POS_Y',  val_def=5,    type='int')
        self.main_win_width  = self.declareParameter(name='MAIN_WIN_WIDTH',  val_def=1200, type='int')
        self.main_win_height = self.declareParameter(name='MAIN_WIN_HEIGHT', val_def=700,  type='int')

        self.main_vsplitter  = self.declareParameter(name='MAIN_VSPLITTER', val_def=600, type='int')
        self.main_tab_name   = self.declareParameter(name='MAIN_TAB_NAME', val_def='Configuration', type='str')


        self.current_config_tab = self.declareParameter(name='CURRENT_CONFIG_TAB', val_def='Configuration File', type='str')
        self.cdb_host = self.declareParameter(name='CDB_HOST', val_def='psanaphi105', type='str')
        self.cdb_port = self.declareParameter(name='CDB_PORT', val_def=27017, type='int')
        self.cdb_hsplitter = self.declareParameter(name='CDB_HSPLITTER', val_def=250, type='int')
        self.cdb_filter = self.declareParameter(name='CDB_FILTER', val_def='', type='str')

#------------------------------
        
#    def __del__(self) :
#        #from psana.pyalgos.generic.Logger import logger
#        #logger.debug('In d-tor:', self.__class__.__name__)
#        print('In CMConfigParameters d-tor')
#        if self.save_cp_at_exit.value() :
#            self.saveParametersInFile()
#        #ConfigParameters.__del__(self)

#------------------------------

cp = CMConfigParameters()

#------------------------------

def test_CMConfigParameters() :
    from psana.pyalgos.generic.Logger import logger as log

    log.setPrintBits(0o377)
    cp.readParametersFromFile()
    cp.printParameters()
    cp.log_level.setValue('debug')
    cp.saveParametersInFile()

#------------------------------

if __name__ == "__main__" :
    import sys
    test_CMConfigParameters()
    sys.exit(0)

#------------------------------
