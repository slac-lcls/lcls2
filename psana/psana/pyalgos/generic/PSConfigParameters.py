#------------------------------
"""
:py:class:`PSConfigParameters` - PS commonly useful configuration parameters
============================================================================

Usage ::
    # Import
    from psana.pyalgos.generic.PSConfigParameters import PSConfigParameters

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
#------------------------------

from psana.pyalgos.generic.ConfigParameters import ConfigParameters

#------------------------------

class PSConfigParameters(ConfigParameters) :
    """A storage of configuration parameters for Experiment Monitor (EM) project.
    """
    char_expand    = u' \u25BC' # down-head triangle
    char_shrink    = u' \u25B2' # solid up-head triangle
 
    list_of_instr = ['AMO', 'SXR', 'XPP', 'XCS', 'CXI', 'MEC', 'MFX', 'DIA', 'MOB', 'USR']
    list_of_dsext = ['None', 'smd', 'smd:live', 'shmem']
    #list_of_dsext = ['None', 'idx', 'smd', 'smd:live', 'shmem']
                    #See: https://confluence.slac.stanford.edu/display/PSDM/Real-Time+Analysis

    def __init__(self, fname=None) :
        """fname : str - the file name with configuration parameters, if not specified then use default.
        """
        #self._name = self.__class__.__name__
        #log.debug('In %s c-tor' % self._name) #, self._name)

        ConfigParameters.__init__(self)
        self._name = 'PSConfigParameters'

        self.declareBaseParameters()

        #print'In PSConfigParameters c-tor started by %s' % __name__
        if  __name__ == '__main__' :
            #self.fname_cp = '%s/%s' % (os.path.expanduser('~'), '.confpars-montool.txt') # Default config file name
            self.fname_cp = './confpars-def.txt' # Default config file name
            self.readParametersFromFile()

#------------------------------
        
    def declareBaseParameters(self) :
        """Declaration of common paramaters for all PS apps"""
        self.list_of_sources = None # for interaction with expmon.PSUtils.list_of_sources() 

        self.instr_dir       = self.declareParameter(name='INSTRUMENT_DIR',  val_def='/reg/d/psdm', type='str') 
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
        self.log_level       = self.declareParameter(name='LOG_LEVEL_OF_MSGS',val_def='info',      type='str' ) 
        self.log_file        = self.declareParameter(name='LOG_FILE_FOR_LEVEL', val_def='./log_for_level.txt', type='str' )
        self.save_cp_at_exit = self.declareParameter(name='SAVE_CONFIG_AT_EXIT', val_def=True, type='bool')

#------------------------------
        
#    def __del__(self) :
#        #from psana.pyalgos.generic.Logger import logger
#        #logger.debug('In d-tor:', self.__class__.__name__)
#        print('In PSConfigParameters d-tor')
#        if self.save_cp_at_exit.value() :
#            self.saveParametersInFile()
#        #ConfigParameters.__del__(self)

#------------------------------

# creation of singleton should be done in the derived class
# cpb = PSConfigParameters()

#------------------------------

def test_PSConfigParameters() :
    #from expmon.Logger import log
    #log.setPrintBits(0377)

    cpb = PSConfigParameters()
    cpb.readParametersFromFile()
    cpb.printParameters()
    cpb.saveParametersInFile()

#------------------------------

if __name__ == "__main__" :
    import sys
    test_PSConfigParameters()
    sys.exit(0)

#------------------------------
