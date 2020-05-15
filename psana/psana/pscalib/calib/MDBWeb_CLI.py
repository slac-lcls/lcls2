#------------------------------
"""
Created on 2020-05-15 by Mikhail Dubrovin
"""
#------------------------------

from psana.pscalib.calib.MDB_CLI import *

logger = logging.getLogger(__name__)

#------------------------------

class MDBWeb_CLI(MDB_CLI):

    def __init__(self, parser): 
        MDB_CLI.__init__(self, parser)

    def _warning(self): logger.warning('MDBWeb_CLI: TBD for mode: %s' % self.mode)

    def print_content(self): self._warning()

    def deldoc(self): self._warning()

    def delcol(self): self._warning()

    def deldb(self): self._warning()

    def add(self): self._warning()

    def get(self): self._warning()

    def test(self): logger.warning('MDBWeb_CLI.test')

    def dispatcher(self):
        mode = self.mode
        logger.debug('Mode: %s' % mode)
        if   mode == 'print' : self.print_content()
        elif mode == 'deldoc': self.deldoc()
        elif mode == 'delcol': self.delcol()
        elif mode == 'deldb' : self.deldb()
        elif mode == 'add'   : self.add()
        elif mode == 'get'   : self.get()
        elif mode == 'test'  : self.test()
        else : logger.warning('Non-implemented command mode "%s"\n  Known modes: %s' % (mode,', '.join(MODES)))

#------------------------------

def cdb_web(parser):
    """Calibration Data Base Command Line Interface
    """
    MDBWeb_CLI(parser)

#------------------------------

if __name__ == "__main__":
    import sys
    sys.exit('Run command cdb -w ...')

#------------------------------
