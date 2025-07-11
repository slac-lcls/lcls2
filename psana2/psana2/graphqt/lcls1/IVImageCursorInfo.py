#!@PYTHON@
"""
Class :py:class:`IVImageCursorInfo` is a QWidget for interactive image
======================================================================

QWidget shows cursor coordinates and hovered pixel intensity.

Usage ::
    from PyQt4 import QtGui, QtCore
    from graphqt.IVImageCursorInfo import IVImageCursorInfo

    app = QtGui.QApplication(sys.argv)
    w = IVImageCursorInfo(parent=None)
    w.setWindowTitle('Cursor info')
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

Created on February 1, 2017 by Mikhail Dubrovin
"""
#------------------------------

from PyQt4 import QtGui, QtCore

from graphqt.Logger import log
from graphqt.Styles import style

class IVImageCursorInfo(QtGui.QWidget) :

    def __init__(self, parent=None) :
        QtGui.QWidget.__init__(self, parent=None)
        self._name = self.__class__.__name__

        self.edi = QtGui.QTextEdit('Cursor info box for image')

        self.hbox = QtGui.QHBoxLayout() 
        self.hbox.addWidget(self.edi) 
        self.hbox.addStretch(1)
        self.setLayout(self.hbox) 

        self.set_tool_tips()
        self.set_style()

        #self.connect(self.but_save,  QtCore.SIGNAL('clicked()'), self.on_but)
        #self.connect(self.but_reset, QtCore.SIGNAL('clicked()'), self.on_but)

#------------------------------

    def set_tool_tips(self) :
        self.setToolTip('Cursor position and value on image') 

#------------------------------

    def set_style(self) :
        self.edi.setFixedHeight(40) 
        self.edi.setMinimumWidth(300) 
        #self.edi.setFixedSize(60,30)
        #self.setMinimumSize(40,150) 
        #self.setGeometry(50, 50, 500, 300)
        self.setContentsMargins(-9,-9,-9,-9)
        self.edi.setStyleSheet(style.styleEditInfo + 'font-size: 16pt; font-family: Courier; font-weight: bold;')
        #self.edi.setVisible(self.show_buts)

#------------------------------

    def set_cursor_pos_value(self, ix, iy, v) :
        s = 'x=%d y=%d v=%.1f' % (ix, iy, v)
        self.edi.setText(s)
        #log.info('set_cursor_pos_value %s' % (self._name, s)

#------------------------------

#    def on_but(self) :
#        but = self.but_save if self.but_save.hasFocus()\
#              else self.but_reset  if self.but_reset.hasFocus()\
#              else None
#        #print str(but.text())
#        log.info('%s.on_but %s' % (self._name, str(but.text())))

#------------------------------

if __name__ == "__main__" :
    import sys
    log.setPrintBits(0377) 
    app = QtGui.QApplication(sys.argv)
    w = IVImageCursorInfo(parent=None)
    w.setWindowTitle('Cursor info')
    w.show()
    app.exec_()

#------------------------------
