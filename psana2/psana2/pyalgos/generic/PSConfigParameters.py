
"""
:py:class:`PSConfigParameters` - PS commonly useful configuration parameters
============================================================================

Usage ::
    # Import
    from psana2.pyalgos.generic.PSConfigParameters import PSConfigParameters

    # Methods - see test

See:
    * :py:class:`psana.pyalgos.generic.PSConfigParameters`
    * :py:class:`psana.pyalgos.generic.ConfigParameters`
    * :py:class:`psana.pyalgos.generic.Logger`

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2016-11-22 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-16
"""

from psana2.pyalgos.generic.ConfigParameters import ConfigParameters


class PSConfigParameters(ConfigParameters):
    """A storage of configuration parameters for Experiment Monitor (EM) project.
    """
    char_expand    = u' \u25BC' # down-head triangle
    char_shrink    = u' \u25B2' # solid up-head triangle

    list_of_instr = ['MEC', 'TMO', 'MOB', 'MFX', 'RIX', 'USR', 'SXR', 'AMO', 'XPP', 'CXI', 'DET', 'DIA', 'TST', 'XCS']
    list_of_dsext = ['None', 'smd', 'smd:live', 'shmem'] #See: https://confluence.slac.stanford.edu/display/PSDM/Real-Time+Analysis

    def __init__(self, fname=None):
        """fname: str - the file name with configuration parameters, if not specified then use default.
        """
        #self._name = self.__class__.__name__
        #log.debug('In %s c-tor' % self._name) #, self._name)

        ConfigParameters.__init__(self)
        self._name = 'PSConfigParameters'

        self.declareBaseParameters()

        if __name__ == '__main__':
            #self.fname_cp = '%s/%s' % (os.path.expanduser('~'), '.confpars-montool.txt') # Default config file name
            self.fname_cp = './confpars-def.txt' # Default config file name
            self.readParametersFromFile()


    def declareBaseParameters(self):
        """Declaration of common paramaters for all PS apps"""
        self.list_of_sources = None # for interaction with expmon.PSUtils.list_of_sources() 
        self.instr_dir       = self.declareParameter(name='INSTRUMENT_DIR',  val_def='/cds/data/psdm', type='str') #/reg/d/psdm
        self.instr_name      = self.declareParameter(name='INSTRUMENT_NAME', val_def='SXR',         type='str')
        self.exp_name        = self.declareParameter(name='EXPERIMENT_NAME', val_def='Select',      type='str') # sxr12316'
        self.str_runnum      = self.declareParameter(name='STR_RUN_NUMBER',  val_def='Select',      type='str')
        self.calib_dir       = self.declareParameter(name='CALIB_DIRECTORY', val_def='./calib',     type='str') # './calib'
        self.data_source     = self.declareParameter(name='DATA_SOURCE',     val_def='None',        type='str') # 'cspad'
        self.dsextension     = self.declareParameter(name='DSET_EXTENSION',  val_def='None',        type='str') # 'shmod'
        self.event_number    = self.declareParameter(name='EVENT_NUMBER',    val_def=0,             type='int')
        self.event_step      = self.declareParameter(name='EVENT_STEP',      val_def=1,             type='int')
        self.wait_msec       = self.declareParameter(name='EVENT_DELAY_MSEC',val_def=500,           type='int')
        self.nevents_update  = self.declareParameter(name='EVENTS_UPDATE',   val_def=100,           type='int')
        self.log_level       = self.declareParameter(name='LOG_LEVEL_OF_MSGS',val_def='DEBUG',       type='str' ) 
        self.log_file        = self.declareParameter(name='LOG_FILE_FOR_LEVEL', val_def='./log_for_level.txt', type='str' )
        self.log_prefix      = self.declareParameter(name='LOG_FILE_PREFIX', val_def='/cds/data/psdm/logs/calibman/lcls2', type='str')
        self.save_log_at_exit= self.declareParameter(name='SAVE_LOG_AT_EXIT', val_def=False, type='bool')
        self.save_cp_at_exit = self.declareParameter(name='SAVE_CONFIG_AT_EXIT', val_def=True, type='bool')


def test_PSConfigParameters():
    cpb = PSConfigParameters()
    cpb.readParametersFromFile()
    cpb.printParameters()
    cpb.saveParametersInFile()


if __name__ == "__main__":
    import sys
    test_PSConfigParameters()
    sys.exit(0)

# EOF
