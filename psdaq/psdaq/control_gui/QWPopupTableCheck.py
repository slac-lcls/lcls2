
"""
:py:class:`QWPopupTableCheck` - Popup table of str items and/with check-boxes
==================================================================================

Usage::

    # Test: python lcls2/psdaq/psdaq/control_gui/QWPopupTableCheck.py

    # Import
    from psdaq.control_gui.QWPopupTableCheck import QWPopupTableCheck

    # Methods - see test

See:
    - :py:class:`QWPopupTableCheck`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2/psdaq/psdaq/control_gui/>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

    QWPopupTableCheck < QWidget  <+(has-a) CGWPartitionTable < QWTableOfCheckBoxes < QWTable < QTableView

Created on 2019-03-29 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QSizePolicy
from PyQt5.QtCore import Qt
from psdaq.control_gui.Styles import style
from psdaq.control_gui.CGWPartitionTable import CGWPartitionTable
from psdaq.control_gui.CGJsonUtils import get_platform, set_platform, list_active_procs
from psdaq.control_gui.CGDaqControl import daq_control


class QWPopupTableCheck(QWidget):

    def __init__(self, **kwargs):
        parent = kwargs.get('parent', None)
        QWidget.__init__(self, parent)

        self.kwargs = kwargs
        self.list2d_out = []

        win_title = kwargs.get('win_title', None)
        if win_title is not None : self.setWindowTitle(win_title)

        self.wtab = CGWPartitionTable(**kwargs)

        self.do_ctrl  = kwargs.get('do_ctrl', True)
        self.do_frame = kwargs.get('do_frame', True)

        self.but_apply = QPushButton('&Apply')
        self.but_apply.clicked.connect(self.onApply)

        self.hbox = QHBoxLayout()
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.but_apply)
        self.hbox.addStretch(1)

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.wtab)
        self.vbox.addLayout(self.hbox)
        self.setLayout(self.vbox)

        self.setIcons()
        self.set_style()

    def set_style(self):
        #if not self.do_frame:
        #   self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        styleGray = "background-color: rgb(230, 240, 230); color: rgb(0, 0, 0);" # Gray
        styleDefault = ""
        self.setStyleSheet(styleDefault)

        self.layout().setContentsMargins(0,0,0,5)

        wtab = self.wtab
        wtab.set_style()
        #htab = min(wtab.verticalHeader().length()+wtab.horizontalHeader().height()+5, 600)
        #wtab.setMaximumHeight(htab)
        #self.setMaximumHeight(wtab.height()+20)
        self.setMaximumHeight(1000)
        self.setMaximumWidth(wtab.width()+100)

        self.but_apply.setFixedWidth(70)
        self.but_apply.setFixedHeight(24)

        #self.but_cancel.setFocusPolicy(Qt.NoFocus)
        self.but_apply.setStyleSheet(styleGray)
        self.but_apply.setEnabled(self.do_ctrl)
        self.but_apply.setFlat(not self.do_ctrl)

        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

        #self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint)

    def setIcons(self):
        try :
          from psdaq.control_gui.QWIcons import icon
          icon.set_icons()
          self.but_apply .setIcon(icon.icon_button_ok)
        except : pass

    #def on_but_update(self):
    def update_partition_table(self):
        logger.debug('update_partition_table')
        _, list2d = get_platform() # [[[True,''], 'test/19670/daq-tst-dev02', 'testClient2b'], ...]
        logger.debug('list2d\n',list2d)

        self.kwargs['tableio'] = list2d
        wtab = CGWPartitionTable(**self.kwargs)
        self.vbox.replaceWidget(self.wtab, wtab)
        #self.vbox.removeWidget(self.hbox)
        #self.vbox.addLayout(self.hbox)
        self.wtab.close()
        del self.wtab
        self.wtab = wtab
        self.set_style()

#    def resizeEvent(self, e):
#        #logger.debug('resizeEvent')
#        QWidget.resizeEvent(self, e)
#        self.wtab.set_style()

#def onCancel(self):
#        logger.debug('onCancel')
#        self.reject()

    def onApply(self):
        logger.debug('onApply')
        self.list2d_out = self.wtab.fill_output_object()
        #self.accept()

        dict_platf, list2d = get_platform() # [[[True,''], 'test/19670/daq-tst-dev02', 'testClient2b'], ...]
        set_platform(dict_platf, self.list2d_out)

        ## 2019-03-13 caf: If Select->Apply is successful, an Allocate transition should be triggered.
        ## 2020-07-29 caf: The Allocate transition will update the active detectors file, if necessary.

        list2d_active = list_active_procs(self.list2d_out)

        if len(list2d_active)==0:
            logger.warning('NO PROCESS SELECTED!')
        else:
            daq_control().setState('allocated')


    def table_out(self):
        return self.list2d_out


if __name__ == "__main__" :
    import os
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'

    import sys
    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    app = QApplication(sys.argv)

    title_h = ['sel', 'grp', 'level/pid/host', 'ID']
    tableio = [\
               [[True,  ''], '1', 'drp/123456/drp-tst-dev002', 'cookie_9'],\
               [[True,  ''], '1', 'drp/123457/drp-tst-dev009', 'cookie_1'],\
               [[True,  ''], '1', 'drp/123456/drp-tst-dev003', 'cookie_8'],\
               [[True,  ''], '1', 'drp/123457/drp-tst-dev009', 'cookie_0'],\
               [[False, ''],  '', 'teb/123458/drp-tst-dev001', 'teb1'],\
               [[True,  ''], '1', 'drp/123456/drp-tst-dev004', 'tokie_2'],\
               [[True,  ''], '1', 'drp/123457/drp-tst-dev009', 'tokie_3'],\
               [[True,  ''], '1', 'drp/123456/drp-tst-dev005', 'tokie_4'],\
               [[True,  ''], '1', 'drp/123456/drp-tst-dev006', 'tokie_5'],\
               [[True,  ''], '1', 'drp/123457/drp-tst-dev009', 'tokie_6'],\
               [[True,  ''], '1', 'drp/123456/drp-tst-dev007', 'tokie_8'],\
               [[True,  ''], '1', 'drp/123456/drp-tst-dev008', 'tokie_9'],\
               [[True,  ''], '1', 'drp/123456/drp-tst-dev008', 'tokie_10'],\
               [[True,  ''], '1', 'drp/123456/drp-tst-dev008', 'tokie_11'],\
               [[True,  ''], '1', 'drp/123457/drp-tst-dev009', 'tokie_1'],\
               [[False, ''],  '', 'ctr/123459/drp-tst-acc06',  'control'],\
    ]

    print('%s\nI/O  table:' % (50*'_'))
    for rec in tableio : print(rec)

    w = QWPopupTableCheck(tableio=tableio, title_h=title_h, do_ctrl=True, do_edit=True)
    w.move(200,100)
    w.show()
    app.exec_()

    print('%s\nI/O table:' % (50*'_'))
    for rec in tableio : print(rec)

    print('%s\nOutput table from items:' % (50*'_'))
    for rec in w.table_out() : print(rec)

    del w
    del app

# EOF
