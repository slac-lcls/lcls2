#------------------------------
"""
:py:class:`QWDirName` - widget to enter directory
=================================================

Usage::

    # Import
    from psana.graphqt.QWDirName import QWFileName

    # Methods - see test

See:
    - :py:class:`QWDirName`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2016-12-09 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-15
"""
#------------------------------

from psana.graphqt.QWFileName import *

#------------------------------

class QWDirName(QWFileName) : # QtGui.QWidget
    """Widget for directory name input
    """
    def __init__(self, parent=None, butname='Select', label='Dir:',\
                 path='/reg/neh/home/dubrovin/LCLS/rel-expmon/',\
                 fltr=QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,\
                 show_frame=False) :

        QWFileName.__init__(self, parent, butname, label, path, mode='r', fltr=fltr, show_frame=show_frame)

#------------------------------
 
    def on_but(self):
        logger.debug('on_but')
        path0 = self.edi.text()
        #pdir, dir = os.path.split(path0)
        #pdir, dir = path0.rsplit('/',1)
        path1 = str(QFileDialog.getExistingDirectory(self,'Select directory', path0, self.fltr))

        if   path1 == ''    : return # if nothing is selected
        elif path1 == path0 : return # is selected the same directory
        else :
            self.path = path1
            self.edi.setText(self.path)
            self.path_is_changed.emit(self.path)
            #self.emit(QtCore.SIGNAL('path_is_changed(QString)'), self.path)
            #logger.info('Selected file:\n' + self.path, __name__)
            logger.debug('Selected file: %s' % self.path)

#------------------------------

if __name__ == "__main__" :
    from PyQt5.QtWidgets import QApplication
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = QWDirName(None, butname='Select', label='Dir:', path='/reg/neh/home/dubrovin/LCLS/rel-expmon', show_frame=True)
    w.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.show()
    app.exec_()

#------------------------------
