#------------------------------
"""
Class :py:class:`IVTabFileName` file name control window
========================================================

Usage ::

    import sys
    from PyQt4 import QtGui
    from graphqt.IVTabFileName import IVTabFileName
 
    app = QtGui.QApplication(sys.argv)
    w = IVTabFileName(show_mode=03)
    w.w_fname.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.w_calib.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.show()
    app.exec_()

See:
    - :class:`IVMain`
    - :class:`IVMainTabs`
    - :class:`IVMainButtons`
    - :class:`IVImageCursorInfo`
    - :class:`IVConfigParameters`
    - :class:`IVTabDataControl`
    - :class:`IVTabFileName`
    - `graphqt documentation <https://lcls-psana.github.io/graphqt/py-modindex.html>`_.

Created on 2017-02-18 by Mikhail Dubrovin
"""
#------------------------------
import sys
import os
from PyQt4 import QtGui, QtCore

from graphqt.Logger             import log
from graphqt.IVConfigParameters import cp
from graphqt.QWDirName          import QWDirName
from graphqt.QWFileName         import QWFileName
from graphqt.Styles             import style
from graphqt.Frame              import Frame

#------------------------------

#class IVTabFileName(QtGui.QWidget) :
class IVTabFileName(Frame) :
    """ File name input GUI
    """
    def __init__(self, parent=None, show_mode=01) :
        Frame.__init__(self, parent, mlw=1)
        #QtGui.QWidget.__init__(self, parent=None)
        self._name = self.__class__.__name__

        self.show_mode = show_mode
        self.parent = parent

        self.calib_dir = cp.calib_dir
        self.fname_img = cp.fname_img

        #self.lab_ins = QtGui.QLabel('Ins:')
        #self.but_ins = QtGui.QPushButton(self.instr_name.value()) # + self.char_expand)

        self.w_fname = QWFileName(None, butname='Select', label='File:',\
                                  path=self.fname_img.value(), 
                                  fltr='*.txt *.bin *.npy *.h5 *.dat *.data\n *',\
                                  show_frame=False)

        self.w_calib = QWDirName(None, butname='Select', label='Clb:',\
                                 path=self.calib_dir.value(), show_frame=False)

        self.box = QtGui.QVBoxLayout()
        self.box.addWidget(self.w_fname)
        self.box.addWidget(self.w_calib)
        #self.box.addStretch(1)

        self.setLayout(self.box)

        self.set_style()
        self.set_tool_tips()

        self.w_fname.connect_path_is_changed_to_recipient(self.on_but_fname)
        self.w_calib.connect_path_is_changed_to_recipient(self.on_but_calib)
        #self.connect(self.but_run, QtCore.SIGNAL('clicked()'), self.on_but_run)

        if cp.ivmain is None : return
        self.w_fname.connect_path_is_changed_to_recipient(cp.ivmain.on_image_file_is_changed)


    def set_tool_tips(self):
        self.setToolTip('Image file name selection')


    def set_style(self):
        self.w_fname .setVisible(self.show_mode & 1)
        self.w_calib.setVisible(self.show_mode & 2)
        self.w_fname.lab.setStyleSheet(style.styleLabel)
        self.w_calib.lab.setStyleSheet(style.styleLabel)

        #self.setContentsMargins(QtCore.QMargins(-5,-5,-5,-5))
        #self.setContentsMargins(QtCore.QMargins(-9,-9,-9,-9))

        #self.setMinimumWidth(500)
        #self.setGeometry(10, 25, 400, 600)
        #self.setFixedHeight(100)
        #self.w_calib.setMinimumWidth(280)
        #self.lab_ins.setStyleSheet(style.styleLabel)


    def on_but_fname(self, fname):
        w = self.w_fname
        print '%s.%s: fname=%s' % (self._name, sys._getframe().f_code.co_name, fname)
        if not os.path.exists(fname) :
            log.warning('DOES NOT EXIST: %s'%(fname), self._name)
            w.edi.setStyleSheet(style.styleButtonBad)
            w.but.setStyleSheet(style.styleButtonGood)
            return

        w.edi.setStyleSheet(style.styleWhiteFixed)
        w.but.setStyleSheet(style.styleButton)
        par = self.fname_img
        #if fname != par.value() :
        #    self.emit(QtCore.SIGNAL('new_image_file_name_is_selected(QString)'), fname)
        #    #w.edi.setText(par.value())
        par.setValue(fname)


    def on_but_calib(self, cdir):
        w = self.w_calib
        if str(cdir).rsplit('/',1)[1] != 'calib' :
            log.warning('NOT A calib DIRECTORY: %s'%(cdir), self._name)
            w.edi.setStyleSheet(style.styleButtonBad)
            w.but.setStyleSheet(style.styleButtonGood)
            return

        w.edi.setStyleSheet(style.styleWhiteFixed)
        w.but.setStyleSheet(style.styleButton)
        par = self.calib_dir
        par.setValue(cdir)
        #w.edi.setText(par.value())


    def closeEvent(self, e):
        log.debug('%s.closeEvent' % self._name)
        try : self.w_fname.disconnect_path_is_changed_from_recipient(cp.ivmain.on_image_file_is_changed)
        except : pass
        QtGui.QWidget.closeEvent(self, e)


    def test_signal_reception(self, s) :
        print '%s.%s: str=%s' % (self._name, sys._getframe().f_code.co_name, s)

#------------------------------
#------------------------------
#------------------------------
#------------------------------

if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    w = IVTabFileName(show_mode=03)
    w.move(QtCore.QPoint(50,50))
    w.setWindowTitle(w._name)
    w.w_fname.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.w_calib.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.show()
    app.exec_()

#------------------------------
