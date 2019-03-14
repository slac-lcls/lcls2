#--------------------
"""
:py:class:`CGWMainPartition` - widget for control_gui
============================================================================================

Usage::

    # Import
    from psdaq.control_gui.CGWMainPartition import CGWMainPartition

    # Methods - see test

See:
    - :py:class:`CGWMainPartition`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-01-25 by Mikhail Dubrovin
"""
#--------------------

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QGroupBox, QPushButton, QHBoxLayout # , QWidget,  QLabel, QLineEdit, QFileDialog
from PyQt5.QtCore import QPoint # pyqtSignal, Qt, QRectF, QPointF, QTimer

#from psdaq.control_gui.CGWPartitionSelection import CGWPartitionSelection
from psdaq.control_gui.QWDialog import QDialog, QWDialog
from psdaq.control_gui.CGDaqControl import daq_control #, DaqControl #, worker_set_state

from psdaq.control_gui.CGJsonUtils import get_platform, set_platform
from psdaq.control_gui.QWPopupCheckDict import QWPopupCheckDict
from psdaq.control_gui.CGWPartitionDisplay import CGWPartitionDisplay

#--------------------

#class CGWMainPartition(QWidget) :
class CGWMainPartition(QGroupBox) :
    """
    """
    def __init__(self, parent=None):

        QGroupBox.__init__(self, 'Partition', parent)

        #self.parent_ctrl = parent_ctrl

        #self.dict_procs = {'string1':True, 'string2':False, 'string3':True, 'string4':False}
        self.dict_procs = {}

        self.but_roll_call = QPushButton('Roll call')
        self.but_select    = QPushButton('Select')
        self.but_display   = QPushButton('Display')

        self.hbox = QHBoxLayout() 
        self.hbox.addWidget(self.but_roll_call)
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.but_select)
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.but_display)
        self.setLayout(self.hbox)

        self.set_tool_tips()
        self.set_style()

        self.but_roll_call.clicked.connect(self.on_but_roll_call)
        self.but_select.clicked.connect(self.on_but_select)
        self.but_display.clicked.connect(self.on_but_display)

        self.w_select = None
        self.w_display = None
        self.state = None

#--------------------

    def set_tool_tips(self) :
        self.setToolTip('Partition GUI')
        self.but_select.setToolTip('Click on button.') 
        self.but_roll_call.setToolTip('Submits "plat" command.')

#--------------------

    def set_style(self) :
        from psdaq.control_gui.Styles import style
        self.setStyleSheet(style.qgrbox_title)

        #self.setWindowTitle('File name selection widget')
        #self.setMinimumWidth(300)
        #self.edi.setMinimumWidth(210)
        #self.setFixedHeight(34) # 50 if self.show_frame else 34)
        #if not self.show_frame : 
        #self.layout().setContentsMargins(0,0,0,0)

        #style = "background-color: rgb(255, 255, 220); color: rgb(0, 0, 0);" # Yellowish
        #style = "background-color: rgb(100, 240, 200); color: rgb(0, 0, 0);" # Greenish
        #style = "background-color: rgb(255, 200, 220); color: rgb(0, 0, 0);" # Pinkish
        #style = "background-color: rgb(240, 240, 100); color: rgb(0, 0, 0);" # YellowBkg
        #self.setStyleSheet(style)

        #self.setMinimumSize(725,360)
        #self.setFixedSize(750,270)
        #self.setMaximumWidth(800)
 
#--------------------
 
    def on_but_select(self):
        logger.debug('on_but_select')

        #dict_procs = self.dict_procs
        dict_platf, dict_procs = get_platform()

        logger.debug('List of processes:')
        for name,state in dict_procs.items() :
            logger.debug('%s is %s selected' % (name.ljust(10), {False:'not', True:'   '}[state]))

        w = QWPopupCheckDict(None, dict_procs, enblctrl=(self.state=='UNALLOCATED'))
        w.move(self.pos()+QPoint(self.width()/2,200))
        w.setWindowTitle('Select partitions')
        resp=w.exec_()

        logger.debug('resp: %s' % {QDialog.Rejected:'Rejected', QDialog.Accepted:'Accepted'}[resp])

        if resp!=QDialog.Accepted : return

        self.dict_procs = dict_procs
        if self.w_display is not None : 
           self.w_display.fill_list_model(listio=self.list_active_processes())

        set_platform(dict_platf, dict_procs)

        # 2019-03-13 caf: If Select->Apply is successful, an Allocate transition should be triggered.
        #self.parent_ctrl....
        daq_control().setState('allocated')

#--------------------

    def list_active_processes(self):
        if len(self.dict_procs) == 0 :
            logger.warning('list of active processes is empty... Click on "Select" button.')
        return [k for k,v in self.dict_procs.items() if v]

#--------------------
 
    def on_but_display(self):
        logger.debug('on_but_display')
        if  self.w_display is None :
            _, self.dict_procs = get_platform()

            listap = self.list_active_processes()
            self.w_display = CGWPartitionDisplay(parent=None, listio=listap)
                             #CGWPartitionSelection(parent=None, parent_ctrl=self)
            self.w_display.move(self.pos() + QPoint(self.width()+30, 200))
            self.w_display.setWindowTitle('Selected partitions')
            self.w_display.show()
        else :
            self.w_display.close()
            self.w_display = None

#--------------------
 
    def on_but_roll_call(self) :
        """Equivalent to CLI: daqstate -p6 --transition plat
           https://github.com/slac-lcls/lcls2/blob/collection_front/psdaq/psdaq/control/daqstate.py
        """
        logger.debug('on_but_roll_call - command to set transition "plat"')
        rv = daq_control().setTransition('plat')
        if rv is not None : logger.error('Error: %s' % rv)

#--------------------

    def set_buts_enable(self, s) :
        logger.debug('set_buts_enable for state %s' % s)
        self.state = state = s.upper()
        self.but_roll_call.setEnabled(state in ('RESET', 'UNALLOCATED'))
        self.but_select.setEnabled(not(state in ('RESET',)))
        self.but_display.setEnabled(not(state in ('RESET','UNALLOCATED')))

#--------------------

if __name__ == "__main__" :

    import sys
    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = CGWMainPartition(None)
    #w.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.show()
    app.exec_()

#--------------------
