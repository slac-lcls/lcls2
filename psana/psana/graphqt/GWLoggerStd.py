
"""
:py:class:`GWLoggerStd` - GUI for python logger
===============================================

Usage::
    # Test: python lcls2/psana/psana/graphqt/GWLoggerStd.py

    # Import
    from psana.graphqt.GWLoggerStd import GWLoggerStd

    # Methods - see test

See:
    - :py:class:`GWLoggerStd`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2018-04-11 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger() # need in root to intercept messages from all other loggers
#logger = logging.getLogger(__name__)

import os
import sys
from random import randint

from PyQt5.QtWidgets import QApplication, QWidget, QTextEdit, QLabel, QPushButton, QComboBox,\
                            QHBoxLayout, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QTextCursor
import psana.pyalgos.generic.Utils as gu
from psana.graphqt.Styles import style

scrname = sys.argv[0].rsplit('/')[-1]

class QWFilter(logging.Filter):
    def __init__(self, qwlogger):
        #logging.Filter.__init__(self)#, name='')
        self.qwl = qwlogger

    def filter(self, rec):
        msg = self.qwl.formatter.format(rec)
        self.qwl.append_qwlogger(msg)
        #self.print_filter_attributes(rec)
        return True

    def print_filter_attributes(self, rec):
        logger.debug('type(rec): %s'%type(rec))
        logger.debug('dir(rec): %s'%dir(rec))
        logger.debug('dir(logger): %s'%dir(logger))
        #logger.debug('dir(syslog): %s'%dir(self.syslog))
        logger.debug(rec.created, rec.name, rec.levelname, rec.msg)


class GWLoggerStd(QWidget):

    #_name = 'GWLoggerStd'

    def __init__(self, **kwa):

        QWidget.__init__(self, parent=None)

        self.show_buttons = kwa.get('show_buttons', False)
        self.log_level    = kwa.get('log_level', 'INFO')
        self.log_prefix   = kwa.get('log_prefix', './')
        self.log_file     = kwa.get('logfname', 'log-med.txt')
        self.save_log_at_exit = kwa.get('save_log_at_exit', False)

        logger.debug('logging._levelToName: ', logging._levelToName) # {0: 'NOTSET', 50: 'CRITICAL', 20: 'INFO',...
        logger.debug('logging._nameToLevel: ', logging._nameToLevel) # {'NOTSET': 0, 'ERROR': 40, 'WARNING': 30,...

        self.dict_level_to_name = logging._levelToName
        self.dict_name_to_level = logging._nameToLevel
        self.level_names = list(logging._levelToName.values())

        self.edi_txt   = QTextEdit('Logger window')
        self.lab_level = QLabel('Log level:')
        self.but_close = QPushButton('&Close')
        self.but_save  = QPushButton('&Save log-file')
        self.but_rand  = QPushButton('&Random')
        self.cmb_level = QComboBox(self)
        self.cmb_level.addItems(self.level_names)
        self.cmb_level.setCurrentIndex(self.level_names.index(self.log_level))

        self.hboxM = QHBoxLayout()
        self.hboxM.addWidget(self.edi_txt)

        self.hboxB = QHBoxLayout()
        self.hboxB.addStretch(4)
        self.hboxB.addWidget(self.lab_level)
        self.hboxB.addWidget(self.cmb_level)
        self.hboxB.addWidget(self.but_rand)
        self.hboxB.addStretch(1)
        self.hboxB.addWidget(self.but_save)
        self.hboxB.addWidget(self.but_close)

        self.vbox = QVBoxLayout()
        self.vbox.addLayout(self.hboxM)
        self.vbox.addLayout(self.hboxB)
        self.setLayout(self.vbox)

        if self.show_buttons: self.connect_buttons()

        self.set_style()
        self.set_tool_tips()

        self.config_logger()


    def config_logger(self):

        self.append_qwlogger('Start logger\nLog file: %s' % self.log_file)

        levname = self.log_level
        level = self.dict_name_to_level[levname] # e.g. logging.DEBUG

        #tsfmt='%Y-%m-%dT%H:%M:%S'
        tsfmt='T%H:%M:%S'
        fmt = '[%(levelname).1s] %(asctime)s %(name)s:L%(lineno)04d %(message)s'
        #fmt = '[%(levelname).1s] %(name)s:L%(lineno)04d %(message)s' if level==logging.DEBUG else\
        #      '[%(levelname).1s] %(asctime)s %(name)s: %(message)s' # %(filename)s

        #sys.stdout = sys.stderr = open('/dev/null', 'w')

        self.formatter = logging.Formatter(fmt, datefmt=tsfmt)

        # TRICK: add filter to handler to intercept ALL messages
        myhandler = None
        if self.save_log_at_exit:
            gu.create_directory(self.log_file, mode=0o2775, umask=0o0, group='ps-users')
            myhandler = logging.FileHandler(self.log_file, 'w')
        else:
            myhandler = logging.StreamHandler()

        myhandler.addFilter(QWFilter(self))
        myhandler.setFormatter(self.formatter)

        logger.handlers.clear()
        logger.addHandler(myhandler)

        self.set_level(levname) # pass level name
        logger.info('log file: %s %s SAVED AT EXIT'%\
                    (self.log_file, 'IS' if self.save_log_at_exit else 'IS NOT'))

    def set_level(self, level_name='DEBUG'):
        level = self.dict_name_to_level[level_name]
        logger.setLevel(level)
        logger.info('Set logger level %s' % level_name)
        self.log_level = level_name

    def connect_buttons(self):
        self.but_close.clicked.connect(self.on_but_close)
        self.but_save.clicked.connect(self.on_but_save)
        self.but_rand.clicked.connect(self.on_but_rand)
        self.cmb_level.currentIndexChanged[int].connect(self.on_cmb_level)

    def disconnect_buttons(self):
        self.but_close.clicked.disconnect(self.on_but_close)
        self.but_save.clicked.disconnect(self.on_but_save)
        self.but_rand.clicked.disconnect(self.on_but_rand)
        self.cmb_level.currentIndexChanged[int].disconnect(self.on_cmb_level)

    def set_tool_tips(self):
        self.edi_txt    .setToolTip('Window for log messages')
        self.but_close  .setToolTip('Close this window')
        self.but_save   .setToolTip('Save logger content in file')
        self.but_rand   .setToolTip('Inject random message')
        self.cmb_level  .setToolTip('Select logger level of messages to display')

    def set_style(self):
        self.           setStyleSheet(style.styleBkgd)
        self.lab_level .setStyleSheet(style.styleTitle)
        self.but_close .setStyleSheet(style.styleButton)
        self.but_save  .setStyleSheet(style.styleButton)
        self.but_rand  .setStyleSheet(style.styleButton)
        self.cmb_level .setStyleSheet(style.styleButton)
        self.edi_txt   .setReadOnly(True)
        self.edi_txt   .setStyleSheet(style.styleWhiteFixed)

        self.lab_level .setVisible(self.show_buttons)
        self.cmb_level .setVisible(self.show_buttons)
        self.but_save  .setVisible(self.show_buttons)
        self.but_rand  .setVisible(self.show_buttons)
        self.but_close .setVisible(self.show_buttons)

        self.layout().setContentsMargins(0,0,0,0)
        #self.setMinimumSize(300,50)

    def closeEvent(self, e):
        logger.debug('closeEvent')
        #logger.info('%s.closeEvent' % self._name)
        #self.save_log_total_in_file() # It will be saved at closing of GUIMain
        #logger.addHandler(self.handler)
        #self.handler.close()
        QWidget.closeEvent(self, e)

    def on_but_close(self):
        logger.debug('on_but_close')
        self.close()

    def on_but_save(self):
        logger.debug('on_but_save:')
        self.save_log_in_file()

    def on_but_rand(self):
        levels = self.level_names
        level_name = levels[randint(0, len(levels)-1)]
        self.append_qwlogger('===> Inject in logger random message of level %s' % level_name)
        ind = self.dict_name_to_level[level_name]
        logger.log(ind, 'This is a random message of level %s' % level_name)

    def on_cmb_level(self):
        selected = str(self.cmb_level.currentText())
        msg = 'on_cmb_level set %s %s' % (self.lab_level.text(), selected)
        logger.debug(msg)
        self.set_level(selected)
        self.edi_txt.setText('Start logging messages in GWLoggerStd for level %s' % self.log_level)

    def save_log_in_file(self):
        logger.info('save_log_in_file ' + self.log_file)
        path = str(QFileDialog.getSaveFileName(self,
                                               caption   = 'Select the file to save log',
                                               directory = self.log_file,
                                               filter    = '*.txt'
                                               ))
        if path == '':
            logger.debug('Log saving is cancelled.')
            return
        logger.info('Log saved in: %s' % str(path))

    def append_qwlogger(self, msg='...'):
        self.edi_txt.append(msg)
        self.scrollDown()

    def scrollDown(self):
        #logger.debug('scrollDown')
        self.edi_txt.moveCursor(QTextCursor.End)
        self.edi_txt.repaint()
        #self.raise_()
        #self.edi_txt.update()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    w = GWLoggerStd(show_buttons=True)
    #w.setWindowTitle(w._name)
    w.setGeometry(200, 400, 600, 300)

    from psana.graphqt.QWIcons import icon
    icon.set_icons()
    w.setWindowIcon(icon.icon_logviewer)

    w.show()
    app.exec_()
    sys.exit(0)

# EOF
