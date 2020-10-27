#----
"""
:py:class:`CGWMainInfo` - widget for configuration
===================================================

Usage::

    # Import
    from psdaq.control_gui.CGWMainInfo import CGWMainInfo

    # Methods - see test

See:
    - :py:class:`CGWMainInfo`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2/psdaq/psdaq/control_gui>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2020-10-23 by Mikhail Dubrovin
"""
#----

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QGroupBox, QLabel, QLineEdit, QGridLayout
from psdaq.control_gui.CGConfigParameters import cp
from psdaq.control_gui.Styles import style

from PyQt5.QtCore import Qt

#----

class CGWMainInfo(QGroupBox):
    """
    """
    def __init__(self, parent=None):
        QGroupBox.__init__(self, 'Info', parent)
        cp.cgwmaininfo = self

        self.lab_exp = QLabel('exp:')
        self.lab_run = QLabel('run:')
        self.lab_evt = QLabel('events:')
        self.lab_evd = QLabel('drops:')
        self.edi_exp = QLabel('N/A')
        self.edi_run = QLabel('N/A')
        self.edi_evt = QLabel('N/A')
        self.edi_evd = QLabel('N/A')

        self.grid = QGridLayout()
        self.grid.addWidget(self.lab_exp,      0, 0, 1, 1)
        self.grid.addWidget(self.edi_exp,      0, 1, 1, 1)
        self.grid.addWidget(self.lab_run,      0, 2, 1, 1)
        self.grid.addWidget(self.edi_run,      0, 3, 1, 1)
        self.grid.addWidget(self.lab_evt,      1, 0, 1, 1)
        self.grid.addWidget(self.edi_evt,      1, 1, 1, 1)
        self.grid.addWidget(self.lab_evd,      1, 2, 1, 1)
        self.grid.addWidget(self.edi_evd,      1, 3, 1, 1)
        self.setLayout(self.grid)

        self.set_tool_tips()
        self.set_style()


    def set_tool_tips(self):
        self.setToolTip('Information') 


    def set_style(self):
        self.setStyleSheet(style.qgrbox_title)
        self.layout().setContentsMargins(2,0,2,2)

        for fld in (self.edi_exp, self.edi_run, self.edi_evt, self.edi_evd):
            fld.setAlignment(Qt.AlignLeft)
            fld.setStyleSheet(style.styleBold) #styleDefault

        for fld in (self.lab_exp, self.lab_run, self.lab_evt, self.lab_evd):
            fld.setAlignment(Qt.AlignRight)
            fld.setStyleSheet(style.styleDefault) #styleLabel

        self.set_visible_line2(False)
 

    def set_visible_line2(self, is_visible):
        for fld in (self.edi_evt, self.edi_evd, self.lab_evt, self.lab_evd):
            fld.setVisible(is_visible)


    def update_info(self):
        run_number = cp.s_run_number if cp.s_recording else cp.s_last_run_number
        self.lab_run.setText('run' if cp.s_recording else 'last run')
        self.edi_run.setText(str(run_number))
        self.edi_exp.setText(str(cp.s_experiment_name))


    def closeEvent(self, e):
        logger.debug('CGWMainInfo.closeEvent')
        QGroupBox.closeEvent(self, e)
        cp.cgwmaininfo = None

#----

    if __name__ == "__main__":
    #if True:

      def key_usage(self):
        return 'Keys:'\
               '\n  ESC - exit'\
               '\n  U - update info window'\
               '\n'


      def keyPressEvent(self, e):
        if   e.key() == Qt.Key_Escape:
            self.close()

        elif e.key() == Qt.Key_U:
            cp.s_experiment_name = 'testex123'
            cp.s_run_number = 123
            cp.s_last_run_number = 122
            self.update_info()

        else:
            logger.info(self.key_usage())

#----
 
if __name__ == "__main__":

    logging.basicConfig(format='[%(levelname).1s] %(asctime)s %(name)s %(lineno)d: %(message)s', level=logging.DEBUG)

    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = CGWMainInfo(parent=None)
    w.show()
    logger.info('show window')
    app.exec_()

#----
