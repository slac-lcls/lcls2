
"""
Class :py:class:`CMConfigParameters` supports configuration parameters for application
======================================================================================

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Usage ::
    from psana.graphqt.CMConfigParameters import cp, dir_calib
    cp.readParametersFromFile()
    cp.printParameters()
    cp.log_level.setValue('DEBUG')
    cp.saveParametersInFile()

See:
    - :class:`CMMain`
    - :class:`CMMainTabs`
    - `graphqt documentation <https://lcls-psana.github.io/graphqt/py-modindex.html>`_.

Created on 2016-11-22 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-26 by Mikhail Dubrovin
"""
import os

import logging
logger = logging.getLogger(__name__)

from psana.pyalgos.generic.PSConfigParameters import PSConfigParameters
from psana.pyalgos.generic.Utils import list_of_hosts as lshosts
import psana.pscalib.calib.CalibConstants as cc
import psana.pyalgos.generic.PSUtils as psu # dir_calib, dir_exp
from psana.detector.dir_root import DIR_ROOT

class CMConfigParameters(PSConfigParameters):
    """A storage of configuration parameters for LCLS-2 Calibration Manager (CM)
    """
    char_expand    = u' \u25BC' # down-head triangle
    char_shrink    = u' \u25B2' # solid up-head triangle

    DB_COLS = 1
    DOCS    = 4
    COLS    = 'collections'
    DBS     = 'DBs'
    cc      = cc
    kwargs  = {'webint':True, 'loglevel':'INFO'}

    def __init__(self, fname=None):
        """fname: str - the file name with configuration parameters, if not specified then use default.
        """
        PSConfigParameters.__init__(self)

        logger.debug('In %s c-tor')

        #self.fname_cp = '%s/%s' % (os.path.expanduser('~'), '.confpars-montool.txt') # Default config file name
        self.fname_cp = './cm-confpars.txt' # Default config file name

        self.declareParameters()
        self.readParametersFromFile()
        #self.printParameters()

        self.list_of_hosts = ['psdb01','psdb02','psdb03','psdb04',\
           'psanagpu114', 'psanaphi105', 'psanaphi106', 'psanaphi107','psdbdev01']
        self.list_of_hosts.append('psdb-dev')

        self.list_of_ports = (27017, 27018, 27019, 27020, 27021, 9306, 9307, 9308)
        self.list_of_str_ports = [str(v) for v in self.list_of_ports]

        self.list_of_doc_widgets = ('Text','List','Table')

        # Widgets with direct access
        self.cmwmain     = None
        self.cmwmaintabs = None
        self.cmwdbmain   = None
        self.cmwdbtree   = None
        self.cmwdbdocs   = None
        self.cmwdbdocswidg=None
        self.qwloggerstd = None
        self.cmwdbdoceditor0 = None
        self.cmwdbdoceditor = None
        self.ivspectrum  = None
        self.ivcontrol   = None
        self.ivmain      = None
        self.ivimageaxes = None
        self.wlog        = None

        self.gwimageaxes = None
        self.gwspectrum = None

        self.last_selection = None
        self.user = cc.USERNAME
        self.upwd = None

        self.h5vmain = None
        self.fstree  = None
        self.fmwtabs = None
        self.fmw1main = None
        self.fmw1control = None

        self.dmqwmain = None
        self.dmqwlist = None
        self.dmqwcontrol = None


    def declareParameters(self):
        # Possible typs for declaration: 'str', 'int', 'long', 'float', 'bool'
        self.log_level = self.declareParameter(name='LOG_LEVEL', val_def='DEBUG', type='str') # val_def='NOTSET'

        #self.log_file - DEPRICATED
        self.log_file  = self.declareParameter(name='LOG_FILE_NAME', val_def='cm-log.txt', type='str')
        self.log_prefix = self.declareParameter(name='LOG_FILE_PREFIX', val_def=DIR_ROOT + '/detector/calib2/constants/logs/', type='str')
        self.save_log_at_exit = self.declareParameter(name='SAVE_LOG_AT_EXIT', val_def=True,  type='bool')

        self.main_win_pos_x  = self.declareParameter(name='MAIN_WIN_POS_X',  val_def=5,    type='int')
        self.main_win_pos_y  = self.declareParameter(name='MAIN_WIN_POS_Y',  val_def=5,    type='int')
        self.main_win_width  = self.declareParameter(name='MAIN_WIN_WIDTH',  val_def=800, type='int')
        self.main_win_height = self.declareParameter(name='MAIN_WIN_HEIGHT', val_def=800,  type='int')

        self.main_vsplitter  = self.declareParameter(name='MAIN_VSPLITTER', val_def=600, type='int')
        self.main_tab_name   = self.declareParameter(name='MAIN_TAB_NAME', val_def='CDB', type='str')

        self.current_config_tab = self.declareParameter(name='CURRENT_CONFIG_TAB', val_def='Parameters', type='str')
        self.cdb_host = self.declareParameter(name='CDB_HOST', val_def=cc.HOST, type='str')
        self.cdb_port = self.declareParameter(name='CDB_PORT', val_def=cc.PORT, type='int')
        self.cdb_hsplitter0 = self.declareParameter(name='CDB_HSPLITTER0', val_def=220, type='int')
        self.cdb_hsplitter1 = self.declareParameter(name='CDB_HSPLITTER1', val_def=600, type='int')
        self.cdb_hsplitter2 = self.declareParameter(name='CDB_HSPLITTER2', val_def=0, type='int')
        self.cdb_filter  = self.declareParameter(name='CDB_FILTER', val_def='', type='str')
        self.cdb_buttons = self.declareParameter(name='CDB_BUTTONS', val_def=1+2+8+16+32+128+1024+2048+8192, type='int')
        self.cdb_docw = self.declareParameter(name='CDB_DOC_WIDGET', val_def='List', type='str')
        self.cdb_selection_mode = self.declareParameter(name='CDB_SELECTION_MODE', val_def='extended', type='str')
        self.iv_buttons = self.declareParameter(name='IV_BUTTONS', val_def=2+4+32+64, type='int')

        self.fmwtab_tab_name = self.declareParameter(name='FMWTAB_TAB_NAME', val_def='LCLS1', type='str')
        self.last_selected_fname = self.declareParameter(name='LAST_SELECTED_FNAME', val_def=None, type='str')
        self.last_selected_data = None # not persistent
        self.last_selected_run = None # not persistent

        self.instr_name = self.declareParameter(name='INSTRUMENT_NAME', val_def='XPP',      type='str')
        self.exp_name   = self.declareParameter(name='EXPERIMENT_NAME', val_def='xpptut21', type='str') # 'Select'


cp = CMConfigParameters()


def expname_def():
    return 'xpptut21' if cp.exp_name.is_default() else\
           cp.exp_name.value()

def dir_exp(expname=expname_def()): return psu.dir_exp(expname)
def dir_xtc(expname=expname_def()): return psu.dir_xtc(expname)
def dir_calib(expname=expname_def()): return psu.dir_calib(expname)

def dirs_to_search(expname=expname_def()):
    return [dir_calib(expname), os.getcwd()]# os.path.expanduser('~')


if __name__ == "__main__":
  def test_CMConfigParameters():

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    cp.readParametersFromFile()
    cp.printParameters()
    cp.log_level.setValue('DEBUG')
    cp.saveParametersInFile()


if __name__ == "__main__":
    test_CMConfigParameters()
    sys.exit(0)

# EOF
