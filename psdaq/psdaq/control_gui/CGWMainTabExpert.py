"""
Class :py:class:`CGWMainTabExpert` is a QWidget for interactive image
=======================================================================

Usage ::

    import sys
    from PyQt5.QtWidgets import QApplication
    from psdaq.control_gui.CGWMainTabExpert import CGWMainTabExpert
    app = QApplication(sys.argv)
    w = CGWMainTabExpert(None, app)
    w.show()
    app.exec_()

See:
    - :class:`CGWMainTabExpert`
    - :class:`CGWMainPartition`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2/psdaq/psdaq/control_gui>`_.

Created on 2019-05-07 by Mikhail Dubrovin
"""
#------------------------------

import logging
logger = logging.getLogger(__name__)

#------------------------------

import json
from time import time

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QTextEdit, QSizePolicy
from PyQt5.QtCore import Qt, QSize

from psdaq.control_gui.CGWMainPartition import CGWMainPartition
from psdaq.control_gui.CGWMainControl   import CGWMainControl

#------------------------------

class CGWMainTabExpert(QWidget) :

    _name = 'CGWMainTabExpert'

    def __init__(self, **kwargs) :

        parent      = kwargs.get('parent', None)
        parent_ctrl = kwargs.get('parent_ctrl', None)

        QWidget.__init__(self, parent=None)

        logger.debug('In %s' % self._name)

        self.wpart = CGWMainPartition()
        parent_ctrl.wpart = self.wpart
        parent_ctrl.wcoll = self.wpart.wcoll

        self.wctrl = CGWMainControl(parent, parent_ctrl)
        parent_ctrl.wctrl = self.wctrl 

        #self.wpart = QTextEdit('Txt 1')
        #self.wctrl = QTextEdit('Txt 2')

        self.vspl = QSplitter(Qt.Vertical)
        self.vspl.addWidget(self.wpart) 
        self.vspl.addWidget(self.wctrl) 

        self.mbox = QHBoxLayout() 
        self.mbox.addWidget(self.vspl)
        self.setLayout(self.mbox)

        self.set_style()

#------------------------------

    def set_tool_tips(self) :
        pass
        #self.butStop.setToolTip('Not implemented yet...')

#--------------------

    def sizeHint(self):
        return QSize(300, 280)

#--------------------

    def set_style(self) :
        self.setMinimumSize(280, 260)
        self.layout().setContentsMargins(0,0,0,0)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)

#--------------------

    def closeEvent(self, e) :
        logger.debug('%s.closeEvent' % self._name)

        try :
            pass
            #self.wpart.close()
            #self.wctrl.close()
        except Exception as ex:
            print('Exception: %s' % ex)

#--------------------

    if __name__ == "__main__" :
 
      def resizeEvent(self, e):
        #logger.debug('resizeEvent', self._name) 
        print('CGWMainTabExpert.resizeEvent: %s' % str(self.size()))


     #def moveEvent(self, e) :
        #logger.debug('moveEvent', self._name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position), __name__)       
        #logger.info('CGWMainTabExpert.moveEvent - move window to x,y: ', str(self.mapToGlobal(QPoint(0,0))))
        #self.wimg.move(self.pos() + QPoint(self.width()+5, 0))
        #pass

#--------------------

if __name__ == "__main__" :

    from psdaq.control_gui.CGDaqControl import daq_control, DaqControlEmulator, Emulator
    daq_control.set_daq_control(DaqControlEmulator())

    import sys
    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)

    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    kwargs = {'parent':None, 'parent_ctrl':Emulator()}
    w = CGWMainTabExpert(**kwargs)
    w.show()
    app.exec_()
    del w
    del app

#------------------------------
