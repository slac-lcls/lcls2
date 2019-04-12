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

from PyQt5.QtWidgets import QGroupBox, QPushButton, QHBoxLayout, QDialog
from PyQt5.QtCore import QPoint

from psdaq.control_gui.CGDaqControl import daq_control
from psdaq.control_gui.CGJsonUtils import get_platform, set_platform, list_active_procs
from psdaq.control_gui.QWPopupTableCheck import QWPopupTableCheck, QWTableOfCheckBoxes

#--------------------

#class CGWMainPartition(QWidget) :
class CGWMainPartition(QGroupBox) :
    """
    """
    TABTITLE_H = ['', 'proc/pid/host', 'aliases']

    def __init__(self, parent=None):

        QGroupBox.__init__(self, 'Partition', parent)

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

        dict_platf, list2d = get_platform() # list2d = [[[True,''], 'test/19670/daq-tst-dev02', 'testClient2b'], ...]

        #logger.debug('List of processes:')
        #for rec in list2d :
        #    [[state,name],alias] = rec
        #    logger.debug('%s %s is %s selected' % (name.ljust(10), alias.ljust(10), {False:'not', True:'   '}[state]))

        w = QWPopupTableCheck(tableio=list2d, title_h=self.TABTITLE_H,\
                              do_ctrl=(self.state=='UNALLOCATED'),\
                              win_title='Select partitions',\
                              do_edit=False, is_visv=True, do_frame=True)

        w.setToolTip('Processes control is only available\nin the state UNALLOCATED')
        w.move(self.pos()+QPoint(self.width()/2,200))
        resp=w.exec_()

        logger.debug('resp: %s' % {QDialog.Rejected:'Rejected', QDialog.Accepted:'Accepted'}[resp])

        if resp!=QDialog.Accepted : return

        list2d = w.table_out()

        if self.w_display is not None :
            self.w_display.fill_table_model(tableio=list2d,\
                                            title_h=self.TABTITLE_H,\
                                            do_edit=False, is_visv=False, do_ctrl=False, do_frame=True)

        set_platform(dict_platf, list2d)
        # 2019-03-13 caf: If Select->Apply is successful, an Allocate transition should be triggered.
        #self.parent_ctrl....

        list2d_active = list_active_procs(list2d)
        if len(list2d_active)>0 :
            daq_control().setState('allocated')
        else :
            logger.warning('NO PROCESS SELECTED!')

#--------------------
 
    def on_but_display(self):
        logger.debug('on_but_display')

        if self.w_display is None :
            _, list2d = get_platform() # [[[True,''], 'test/19670/daq-tst-dev02', 'testClient2b'], ...]

            list2d_active = list_active_procs(list2d)
            #logger.debug('list2d active processes:\n%s' % str(list2d_active))

            self.w_display = QWTableOfCheckBoxes(parent=None, tableio=list2d_active,\
                                                 win_title='Display partitions',\
                                                 title_h=self.TABTITLE_H,\
                                                 is_visv=False)

            self.w_display.setToolTip('Processes selection is only available\nin the "Select" window.')
            self.w_display.move(self.pos() + QPoint(self.width()+30, 200))
            self.w_display.setWindowTitle('Display partitions')
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
        """By Chris F. logistics sets buttons un/visible.
        """
        logger.debug('set_buts_enable for state %s' % s)
        self.state = state = s.upper()
        self.but_roll_call.setEnabled(state in ('RESET', 'UNALLOCATED'))
        self.but_select.setEnabled(not(state in ('RESET',)))
        self.but_display.setEnabled(not(state in ('RESET','UNALLOCATED')))

        if state in ('RESET', 'UNALLOCATED') and self.w_display is not None :
            self.w_display.close()
            self.w_display = None

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
