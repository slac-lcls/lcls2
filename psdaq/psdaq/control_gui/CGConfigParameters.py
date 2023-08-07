
"""
Class :py:class:`CGConfigParameters` supports configuration parameters for application
======================================================================================

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Usage ::
    from psdaq.control_gui.CGConfigParameters import cp

    cp.readParametersFromFile()
    cp.printParameters()
    cp.log_level.setValue('DEBUG')
    cp.saveParametersInFile()

See:
    - :class:`CGWMain`
    - :class:`CGWMainControl`
    - `graphqt documentation <https://github.com/slac-lcls/lcls2/tree/master/psdaq/psdaq/control_gui>`_.

Created on 2019-06-10 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

from psdaq.control_gui.ConfigParameters import ConfigParameters

class CGConfigParameters(ConfigParameters) :
    """A storage of configuration parameters for LCLS-2 DAQ Control GUI (CG)"""
    char_expand    = u' \u25BC' # down-head triangle
    char_shrink    = u' \u25B2' # solid up-head triangle

    def __init__(self, fname=None) :
        """fname : str - the file name with configuration parameters, if not specified then use default.
        """
        ConfigParameters.__init__(self)

        logger.debug('In %s c-tor')

        self.fname_cp = './cg-confpars.txt' # Default config file name

        self.declareParameters()
        self.readParametersFromFile()

        # Registration of widgets/objects
        self.qapplication      = None
        self.cgwmain           = None
        self.cgwmaincollection = None
        self.cgwmainpartition  = None
        self.cgwmaincontrol    = None
        self.cgwmaintabuser    = None
        self.cgwmainconfiguration = None
        self.cgwmaininfo       = None
        self.cgwconfigeditor   = None
        self.cgwpartitiontable = None
        self.qwloggererror     = None

        # DAQ status cache
        self.s_transition      = None
        self.s_state           = None
        self.s_cfgtype         = None
        self.s_recording       = None
        self.s_platform        = None
        self.s_experiment_name = None
        self.s_run_number      = None
        self.s_last_run_number = None

        self.instr             = None


    def test_cpinit(self) :
        self.s_transition = 'tst-transition'
        self.s_state      = 'tst-state'
        self.s_cfgtype    = 'tst-cfgtype'
        self.s_recording  = 'tst-recording'
        self.s_platform   = 'tst-platform'


    def declareParameters(self) :

        # Possible typs for declaration : 'str', 'int', 'long', 'float', 'bool'
        self.log_level = self.declareParameter(name='LOG_LEVEL', val_def='INFO', type='str') # val_def='NOTSET'
        self.main_win_pos_x  = self.declareParameter(name='MAIN_WIN_POS_X',  val_def=100, type='int')
        self.main_win_pos_y  = self.declareParameter(name='MAIN_WIN_POS_Y',  val_def=5,   type='int')
        self.main_win_width  = self.declareParameter(name='MAIN_WIN_WIDTH',  val_def=370, type='int')
        self.main_win_height = self.declareParameter(name='MAIN_WIN_HEIGHT', val_def=810, type='int')


cp = CGConfigParameters()


def test_CGConfigParameters() :

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    cp.readParametersFromFile()
    cp.printParameters()
    cp.log_level.setValue('DEBUG')
    cp.saveParametersInFile()


if __name__ == "__main__" :
    import sys
    test_CGConfigParameters()
    sys.exit(0)

# EOF
