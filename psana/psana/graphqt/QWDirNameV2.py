
"""
:py:class:`QWDirNameV2` - widget to enter directory
=================================================

Usage::

    # Import
    from psana.graphqt.QWDirNameV2 import QWDirNameV2

    # Methods - see test

See:
    - :py:class:`QWDirNameV2`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2016-12-09 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-15
"""

from psana.graphqt.QWFileNameV2 import *


class QWDirNameV2(QWFileNameV2):
    """Widget for directory name input
    """
    def __init__(self, parent=None, label='Dir:',\
                 path='/cds/group/psdm',\
                 mode='r',\
                 fltr=QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,\
                 dirs=[os.path.expanduser('~'), './calib'],\
                 hide_path=True):

        QWFileNameV2.__init__(self, parent, label, path, mode, fltr, dirs=dirs, hide_path=hide_path)

 
    def on_but(self):
        #logger.debug('on_but')
        path0 = self.but.text()
        path1 = str(QFileDialog.getExistingDirectory(self,'Select directory', path0, self.fltr))

        if   path1 == '':
            logger.debug('nothing selected')
            return
        elif path1 == path0:
            logger.debug('path has not been changed: %s' % str(path0))
            return
        else:
            self.path = path1
            self.but.setText(self.path)
            self.path_is_changed.emit(self.path)
            logger.debug('selected path: %s' % self.path)


if __name__ == "__main__":
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'
    from PyQt5.QtWidgets import QApplication
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d: %(message)s', level=logging.DEBUG)#%(name)s 
    app = QApplication(sys.argv)
    w = QWDirNameV2(None, label='Dir:', path='/cds/group/psdm/Select')
    w.setGeometry(100, 50, 400, 80)
    w.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.show()
    app.exec_()

# EOF
