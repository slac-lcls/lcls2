#------------------------------
"""SSConfigParameters - class supporting configuration parameters for application.
   Created: 2017-07-26 
   Author : Mikhail Dubrovin
"""
#------------------------------
# import os
from expmon.PSConfigParameters import PSConfigParameters
from expmon.PSNameManager      import nm # It is here for initialization

#------------------------------

class SSConfigParameters(PSConfigParameters) :
    """A storage of configuration parameters for Experiment Monitor (EM) project.
    """

    def __init__(self, fname=None) :
        """fname : str - the file name with configuration parameters, if not specified then use default.
        """
        PSConfigParameters.__init__(self)
        #self._name = self.__class__.__name__
        #log.debug('In c-tor', self._name)
        #print 'SSConfigParameters c-tor'# % self._name

        #self.fname_cp = '%s/%s' % (os.path.expanduser('~'), '.confpars-montool.txt') # Default config file name
        self.fname_cp = './sourse-selector-confpars.txt' # Default config file name
        self.list_of_sources = None # if None - updated in the ThreadWorker
        self.list_of_sources_selected = None
        self.number_of_sources_max = 200

        self.declareParameters()
        self.readParametersFromFile()

        nm.set_config_pars(self)

#------------------------------
        
    def declareParameters(self) :
        # Possible typs for declaration : 'str', 'int', 'long', 'float', 'bool'
        self.log_level = self.declareParameter(name='LOG_LEVEL_OF_MSGS', val_def='info', type='str')
        self.dir_log_repo     = self.declareParameter(name='DIR_LOG_REPO', val_def='/reg/g/psdm/logs/sourse-selector', type='str')
        self.log_file         = self.declareParameter(name='LOG_FILE_NAME', val_def='sourse-selector-log.txt', type='str')
        self.save_log_at_exit = self.declareParameter( name='SAVE_LOG_AT_EXIT', val_def=True,  type='bool')

        #self.main_win_pos_x  = self.declareParameter(name='MAIN_WIN_POS_X',  val_def=5,   type='int')
        #self.main_win_pos_y  = self.declareParameter(name='MAIN_WIN_POS_Y',  val_def=5,   type='int')
        #self.main_win_width  = self.declareParameter(name='MAIN_WIN_WIDTH',  val_def=900, type='int')
        #self.main_win_height = self.declareParameter(name='MAIN_WIN_HEIGHT', val_def=600, type='int')

        # LISTS of parameters

        det_srcs_def = [('None', 'None', 'str') for i in range(self.number_of_sources_max)]
        self.det_src_list = self.declareListOfPars('DET_SRC', det_srcs_def)

#------------------------------

cp = SSConfigParameters()

#------------------------------

def test_SSConfigParameters() :
    from expmon.Logger import log

    log.setPrintBits(0o377)
    cp.readParametersFromFile()
    cp.printParameters()
    cp.log_level.setValue('debug')
    cp.saveParametersInFile()

#------------------------------

if __name__ == "__main__" :
    import sys
    test_SSConfigParameters()
    sys.exit(0)

#------------------------------
