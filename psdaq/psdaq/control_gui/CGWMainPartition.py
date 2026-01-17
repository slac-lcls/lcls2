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

from PyQt5.QtWidgets import QGroupBox, QPushButton, QHBoxLayout, QVBoxLayout, QDialog, QSizePolicy
from PyQt5.QtCore import Qt, QPoint, QSize
from PyQt5.QtGui import QCursor

from psdaq.control_gui.CGDaqControl import daq_control
from psdaq.control_gui.CGJsonUtils import get_platform, set_platform, list_active_procs
from psdaq.control_gui.QWPopupTableCheck import QWPopupTableCheck
from psdaq.control_gui.CGWPartitionTable import CGWPartitionTable

from psdaq.control_gui.CGWMainCollection import CGWMainCollection
from psdaq.control_gui.CGConfigParameters import cp

#--------------------

class CGWMainPartition(QGroupBox):
    """
    """
    TABTITLE_H = ['sel', 'grp', 'level/pid/host', 'ID', 'Monitor']

    def __init__(self, parent=None):

        QGroupBox.__init__(self, 'Partition', parent)
        cp.cgwmainpartition = self

        self.but_select = QPushButton('Select')
        self.but_show   = QPushButton('Show')

        self.wcoll = CGWMainCollection()

        self.hbox = QHBoxLayout() 
        self.hbox.addWidget(self.but_select)
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.but_show)

        self.vbox = QVBoxLayout() 
        self.vbox.addLayout(self.hbox)
        self.vbox.addWidget(self.wcoll)

        self.setLayout(self.vbox)

        self.set_tool_tips()
        self.set_style()

        self.but_select.clicked.connect(self.on_but_select)
        self.but_show.clicked.connect(self.on_but_show)

        self.w_select = None
        self.w_show = None
        self.w_select = None
        self.set_buts_enabled()

        # this line is a useful debugging tool when doing automated
        # testing: it automatically "clicks" the partition-select
        # button which normally needs to be clicked
        # manually. -cpo 01/16/26
        #print('*** hack: autoclick the partition select button')
        #self.but_select.click()

#--------------------

    def set_tool_tips(self):
        self.setToolTip('Partition GUI')
        self.but_select.setToolTip('Select partitions')
        self.but_show.setToolTip('Show partitions')

#--------------------

    def set_style(self):
        from psdaq.control_gui.Styles import style
        self.setStyleSheet(style.qgrbox_title)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
        self.layout().setContentsMargins(4,10,4,4)
        self.setMinimumSize(150,100)

#--------------------

    def sizeHint(self):
        return QSize(200, 120)
 
#--------------------
 
    def on_but_select(self):
        logger.debug('on_but_select')
        if self.w_select is not None:
            self.w_select.close()
            self.w_select = None
        self.roll_call()

#--------------------
 
    def open_select_window(self):
        logger.debug('open_select_window')
        dict_platf, list2d = get_platform() # list2d = [[[True,''], 'test/19670/daq-tst-dev02', 'testClient2b'], ...]

        #logger.debug('List of processes:')
        #for rec in list2d:
        #    [[state,name],alias] = rec
        #    logger.debug('%s %s is %s selected' % (name.ljust(10), alias.ljust(10), {False:'not', True:'   '}[state]))

        #parent=self,

        self.w_select = QWPopupTableCheck(tableio=list2d, title_h=self.TABTITLE_H,\
                              do_ctrl=self.do_ctrl,\
                              win_title='Select partition',\
                              do_edit=False, is_visv=False, do_frame=True)

        if not self.do_ctrl:
            self.w_select.setToolTip('Processes control is only available\nin the state UNALLOCATED or RESET')

        #self.w_select.move(QCursor.pos()+QPoint(20,10))
        self.w_select.move(self.mapToGlobal(self.but_select.pos()) + QPoint(5, 22)) # (5,22) offset for frame
        # this line is a useful debugging tool when using daqstate to
        # transition automatically between states for testing: it automatically
        # does the partition-select "apply" which normally needs to be clicked
        # manually. one does have to remove this line once to manually select
        # the desired detectors, and then add this line back in to be able
        # go from UNALLOCATED to the desired state without requiring any
        # manual clicking. -cpo 01/16/26
        print('*** automation: click apply for partition select')
        self.w_select.but_apply.click()
        resp=self.w_select.show()

        # MOVED TO QWPopupTableCheck

        #resp=self.w_select.exec_()
        #logger.info('resp: %s' % {QDialog.Rejected:'Rejected', QDialog.Accepted:'Accepted'}[resp])
        #if resp!=QDialog.Accepted: return
        #list2d = self.w_select.table_out()

        #self.w_select = None
        #set_platform(dict_platf, list2d)

        ## 2019-03-13 caf: If Select->Apply is successful, an Allocate transition should be triggered.
        ## 2020-07-29 caf: The Allocate transition will update the active detectors file, if necessary.

        #list2d_active = list_active_procs(list2d)

        #if len(list2d_active)==0:
        #    logger.warning('NO PROCESS SELECTED!')

        #daq_control().setState('allocated')

#--------------------

    def update_select_window(self):
        print('*** update select')
        logger.debug('update_select_window')
        if self.w_select is None: return
        print('*** update_part')
        self.w_select.update_partition_table()

#--------------------

    def update_show_window(self):
        logger.debug('update_show_window')
        if cp.cgwpartitiontable is None: self.w_show = None
        if self.w_show is None: return
        _, list2d = get_platform() # list2d = [[[True,''], 'test/19670/daq-tst-dev02', 'testClient2b'], ...]
        self.w_show.fill_table_model(tableio=list2d,\
                                     title_h=self.TABTITLE_H,\
                                     do_edit=False, is_visv=False, do_ctrl=False, do_frame=True)

#--------------------

    def on_but_show(self):
        logger.debug('on_but_show')

        if cp.cgwpartitiontable is None: self.w_show = None
        if self.w_show is None:
            _, list2d = get_platform() # [[[True,''], 'test/19670/daq-tst-dev02', 'testClient2b'], ...]

            list2d_active = list_active_procs(list2d)
            #logger.debug('list2d active processes:\n%s' % str(list2d_active))

            self.w_show = CGWPartitionTable(parent=None, tableio=list2d_active,\
                                            win_title='Display partitions',\
                                            title_h=self.TABTITLE_H,\
                                            is_visv=False)

            self.w_show.setToolTip('Processes selection is only available\nin the "Select" window.')
            self.w_show.move(QCursor.pos()+QPoint(50,10))
            self.w_show.setWindowTitle('Partitions')
            self.w_show.show()

            #if cp.cgwmain is not None:
               # THIS DOES NOT WORK AT LEAST IN OUR WM ...
               #logger.debug('force to activate cgwmain window')
               #cp.cgwmain.setWindowTitle('Activate it')
               #cp.cgwmain.raise_()
               #cp.cgwmain.activateWindow()

            """
               # THIS WAS A FIGHT FOR ACTIVATION OF OTHER WINDOW, BUT
               # ALL THIS DOES NOT WORK AT LEAST IN OUR WM ...
               #cp.cgwmain.setWindowState(cp.cgwmain.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)
               #cp.cgwmain.show()
               #cp.cgwmain.setFocus(Qt.PopupFocusReason)
               #cp.cgwmain.setFocus()
               #cp.cgwmain.update()

            if cp.qapplication is not None:
                for w in cp.qapplication.allWindows():
                    print('window in allWindows():', str(w), type(w), w.title(), ' isActive:', w.isActive(),' isModal:', w.isModal())
                    if str(w.title())=='DAQ Control':
                        print('window %s is found' % w.title())
                        #w.show()
                        w.requestActivate()

            from PyQt5.QtTest import QTest
            QTest.mouseClick(self, Qt.LeftButton)
            """

        else:
            self.w_show.close()
            self.w_show = None

        self.set_but_show_title()


    def set_but_show_title(self):
        """Change but_show title depending on status of the cp.cgwpartitiontable window,
           which can be closed at click on but_show or window [x].
        """
        status = cp.cgwpartitiontable is None
        self.but_show.setText('Show' if status else 'Close')
        self.but_show.setToolTip('Show partitions' if status else 'Close partitions window')

#--------------------
 
    def roll_call(self):
        """Equivalent to CLI: daqstate -p6 --transition plat
           https://github.com/slac-lcls/lcls2/blob/collection_front/psdaq/psdaq/control/daqstate.py
        """
        logger.debug('roll_call - command to set transition "rollcall"')
        rv = daq_control().setTransition('rollcall')
        if rv is not None: logger.error('Error: %s' % rv)

#--------------------

    def set_buts_enabled(self):
        """By Chris F. logistics https://confluence.slac.stanford.edu/display/~caf/DAQ+GUI+Notes
        """
        s = cp.s_state
        trans = cp.s_transition
        logger.debug('transition: %s' % trans)
        logger.info('set_buts_enabled for state "%s"' % s)
        state = s.lower() if s is not None else 'None'
        self.do_ctrl = enabled = (state in ('reset','unallocated'))

        self.but_select.setEnabled(enabled)
        self.but_show.setEnabled(not enabled)

        if cp.cgwpartitiontable is None: self.w_show = None
        if enabled and self.w_show is not None:
            self.w_show.close()
            self.w_show = None

        if trans == 'rollcall':
            if self.w_show   is not None: self.update_show_window()
            if self.w_select is not None: self.update_select_window()

        if state == 'unallocated':
            if self.w_select is None: self.open_select_window()
        else:
           if self.w_select is not None:
              self.w_select.close()
              self.w_select = None

#--------------------

    def closeEvent(self, e):
        logger.debug('closeEvent')
        QGroupBox.closeEvent(self, e)
        cp.cgwmainpartition = None

#--------------------

    if __name__ == "__main__":
 
      def resizeEvent(self, e):
        print('CGWMainPartition.resizeEvent: %s' % str(self.size()))

#--------------------

if __name__ == "__main__":

    from psdaq.control_gui.CGDaqControl import DaqControlEmulator
    daq_control.set_daq_control(DaqControlEmulator())

    import sys
    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = CGWMainPartition(None)
    #w.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.show()
    app.exec_()

#--------------------
