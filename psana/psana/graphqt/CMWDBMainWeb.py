"""
Class :py:class:`CMWDBMainWeb` is a CMWDBMain(QWidget) for calibman with web interface
=======================================================================================

Usage ::
    See test_CMWDBMainWeb() at the end

See:
    - :class:`CMWDBMainWeb`
    - :class:`CMWMainTabs`
    - :class:`CMConfigParameters`
    - `graphqt documentation <https://lcls-psana.github.io/graphqt/py-modindex.html>`_.

Created on 2021-01-12 by Mikhail Dubrovin
"""

from psana.graphqt.CMWDBMain import *

#---

class CMWDBMainWeb(CMWDBMain):

    def __init__(self, parent=None):
        CMWDBMain.__init__(self, parent=parent)
        #self._name = self.__class__.__name__
        self._name = 'CMWDBMainWeb'
        cp.cmwdbmain = self
#---

if __name__ == "__main__":
  def test_CMWDBMain():
    import sys
    from PyQt5.QtWidgets import QApplication
    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = CMWDBMain()
    w.setMinimumSize(600, 300)
    w.show()
    app.exec_()
    del w
    del app

#---

if __name__ == "__main__":
    test_CMWDBMain()

# EOF

