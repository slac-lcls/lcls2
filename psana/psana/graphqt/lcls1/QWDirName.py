#------------------------------
"""
@version $Id: QWDirName.py 12962 2016-12-09 20:06:16Z dubrovin@SLAC.STANFORD.EDU $

@author Mikhail S. Dubrovin
"""
#------------------------------

from graphqt.QWFileName import *

#------------------------------

class QWDirName(QWFileName) : # QtGui.QWidget
    """Widget for directory name input
    """
    def __init__(self, parent=None, butname='Select', label='Dir:',\
                 path='/reg/neh/home/dubrovin/LCLS/rel-expmon/',\
                 fltr=QtGui.QFileDialog.ShowDirsOnly | QtGui.QFileDialog.DontResolveSymlinks,\
                 show_frame=False) :

        QWFileName.__init__(self, parent, butname, label, path, mode='r', fltr=fltr, show_frame=show_frame)
        self._name = self.__class__.__name__

#------------------------------
 
    def on_but(self):
        path0 = self.edi.text()
        #pdir, dir = os.path.split(path0)
        #pdir, dir = path0.rsplit('/',1)
        path1 = str(QtGui.QFileDialog.getExistingDirectory(self,'Select directory', path0, self.fltr))

        if   path1 == ''    : return # if nothing is selected
        elif path1 == path0 : return # is selected the same directory
        else :
            self.path = path1
            self.edi.setText(self.path)
            self.emit(QtCore.SIGNAL('path_is_changed(QString)'), self.path)
            #logger.info('Selected file:\n' + self.path, __name__)
            #print 'Selected file: %s' % self.path

#------------------------------

if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    w = QWDirName(None, butname='Select', label='Dir:', path='/reg/neh/home/dubrovin/LCLS/rel-expmon', show_frame=True)
    w.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.show()
    app.exec_()

#------------------------------
