#------------------------------
"""
:py:class:`QWLoggerStd` - GUI for python logger
===============================================

Usage::
    # Test: python lcls2/psana/psana/graphqt/QWLoggerStd.py

    # Import
    from psana.graphqt.QWLoggerStd import QWLoggerStd

    # Methods - see test

See:
    - :py:class:`QWLoggerStd`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2018-04-11 by Mikhail Dubrovin
"""
#------------------------------
import logging
logger = logging.getLogger() # need in root to intercept messages from all other loggers
#logger = logging.getLogger(__name__)

import os
import sys
from random import randint

from PyQt5.QtWidgets import QWidget, QTextEdit, QLabel, QPushButton, QComboBox,\
                            QHBoxLayout, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QTextCursor
from psana.graphqt.Styles import style

#------------------------------
import psana.pyalgos.generic.Utils as gu

def log_file_name(lfpath) :
    """Returns (str) log file name like /reg/g/psdm/logs/calibman/lcls2/2018/20180518T122407-dubrovin.txt
    """
    t0_sec = gu.time()
    tstamp = gu.str_tstamp('%Y%m%dT%H%M%S', t0_sec) 
    #year_month = gu.str_tstamp('%Y/%m', time_sec=None) 
    year = gu.str_tstamp('%Y', time_sec=None) 
    return '%s/%s/%s-%s.txt' % (lfpath, year, tstamp, gu.get_login())#, os.getpid())

#------------------------------

class QWFilter(logging.Filter) :
    def __init__(self, qwlogger) :
        #logging.Filter.__init__(self)#, name='')
        self.qwl = qwlogger

    def filter(self, rec) :
        msg = self.qwl.formatter.format(rec)
        self.qwl.append_qwlogger(msg)
        #self.print_filter_attributes(rec)
        return True


    def print_filter_attributes(self, rec) :
        logger.debug('type(rec): %s'%type(rec))
        logger.debug('dir(rec): %s'%dir(rec))
        logger.debug('dir(logger): %s'%dir(logger))
        #logger.debug('dir(syslog): %s'%dir(self.syslog))
        logger.debug(rec.created, rec.name, rec.levelname, rec.msg)

#------------------------------
#------------------------------

class QWLoggerStd(QWidget) :

    _name = 'QWLoggerStd'

    def __init__(self, cp, show_buttons=True) :

        QWidget.__init__(self, parent=None)

        self.log_level = cp.log_level
        self.log_prefix  = cp.log_prefix
        self.log_file  = cp.log_file # DEPRICATED

        log_fname = log_file_name(self.log_prefix.value())
        depth = 6 if log_fname[0]=='/' else 1
        gu.create_path(log_fname, depth, mode=0o0777)
        #print('Log file: %s' % log_fname)

        self.show_buttons = show_buttons
        cp.qwloggerstd = self

        #logger.debug('logging.DEBUG: ', logging.DEBUG)
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
        self.cmb_level.setCurrentIndex(self.level_names.index(self.log_level.value()))
        
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

        if self.show_buttons : self.connect_buttons()

        self.set_style()
        self.set_tool_tips()

        self.config_logger(log_fname)


    def config_logger_v0(self, log_fname='cm-log.txt') :
        self.append_qwlogger('Configure logger')

        fmt = '%(asctime)s %(name)s %(levelname)s: %(message)s'
        tsfmt='%Y-%m-%dT%H:%M:%S'

        level = self.dict_name_to_level[self.log_level.value()] # e.g. logging.DEBUG

        self.formatter = logging.Formatter(fmt, datefmt=tsfmt)
        #self.handler = logging.StreamHandler()
        self.handler = logging.FileHandler(log_fname, 'w')
        self.handler.setLevel(logging.NONSET)
        self.handler.addFilter(QWFilter(self))
        self.handler.setFormatter(self.formatter)

        logging.basicConfig(format=fmt,\
                            datefmt=tsfmt,\
                            level=level,\
                            handlers=[self.handler,]
        ) 
        #                    filename=log_fname, filemode='w',\
        ## if filename is not specified - all messages go to sys.tty

        #self.set_level(self.log_level.value()) # pass level name


    def config_logger(self, log_fname='cm-log.txt') :

        self.append_qwlogger('Start logger\nLog file: %s' % log_fname)

        levname = self.log_level.value()
        level = self.dict_name_to_level[levname] # e.g. logging.DEBUG

        tsfmt='%Y-%m-%dT%H:%M:%S'
        fmt = '%(levelname)s %(name)s: %(message)s' if level==logging.DEBUG else\
              '%(asctime)s %(levelname)s %(name)s: %(message)s'

        #sys.stdout = sys.stderr = open('/dev/null', 'w')

        self.formatter = logging.Formatter(fmt, datefmt=tsfmt)
        #logger.addFilter(QWFilter(self)) # register self for callback from filter

        # TRICK: add filter to handler to intercept ALL messages
        #self.handler = logging.StreamHandler()
        self.handler = logging.FileHandler(log_fname, 'w')
        self.handler.addFilter(QWFilter(self))
        #self.handler.setLevel(logging.NOTSET) # level
        self.handler.setFormatter(self.formatter)
        logger.addHandler(self.handler)
        self.set_level(levname) # pass level name

        #logger.debug('dir(self.handler):' , dir(self.handler))


    def set_level(self, level_name='DEBUG') :
        #self.append_qwlogger('Set logger layer: %s' % level_name)
        #logger.setLevel(level_name) # {0: 'NOTSET'}
        level = self.dict_name_to_level[level_name]
        logger.setLevel(level)
        #msg = 'Set logger level %s of the list: %s' % (level_name, ', '.join(self.level_names))
        #logger.debug(msg)
        logger.info('Set logger level %s' % level_name)


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
        #self           .setToolTip('This GUI is for browsing log messages')
        self.edi_txt    .setToolTip('Window for log messages')
        self.but_close  .setToolTip('Close this window')
        self.but_save   .setToolTip('Save logger content in file')#: '+os.path.basename(self.fname_log.value()))
        self.but_rand   .setToolTip('Inject random message')
        self.cmb_level  .setToolTip('Select logger level of messages to display')


    def set_style(self):
        self.           setStyleSheet(style.styleBkgd)
        #self.lab_title.setStyleSheet(style.styleTitleBold)
        self.lab_level .setStyleSheet(style.styleTitle)
        self.but_close .setStyleSheet(style.styleButton)
        self.but_save  .setStyleSheet(style.styleButton) 
        self.but_rand  .setStyleSheet(style.styleButton) 
        self.cmb_level .setStyleSheet(style.styleButton) 
        self.edi_txt   .setReadOnly(True)
        self.edi_txt   .setStyleSheet(style.styleWhiteFixed) 
        #self.edi_txt   .ensureCursorVisible()
        #self.lab_title.setAlignment(QtCore.Qt.AlignCenter)
        #self.titTitle.setBold()

        self.lab_level .setVisible(self.show_buttons)
        self.cmb_level .setVisible(self.show_buttons)
        self.but_save  .setVisible(self.show_buttons)
        self.but_rand  .setVisible(self.show_buttons)
        self.but_close .setVisible(self.show_buttons)

        if not self.show_buttons : self.setContentsMargins(-9,-9,-9,-9)
        #self.setMinimumSize(300,50)
        #self.setBaseSize(500,200)


    #def setParent(self,parent) :
    #    self.parent = parent


    #def resizeEvent(self, e):
        #logger.debug('resizeEvent') 
        #pass


    #def moveEvent(self, e):
        #logger.debug('moveEvent') 
        #self.cp.posGUIMain = (self.pos().x(),self.pos().y())
        #pass


    def closeEvent(self, e):
        logger.debug('closeEvent')
        #logger.info('%s.closeEvent' % self._name)
        #self.save_log_total_in_file() # It will be saved at closing of GUIMain

        #logger.addHandler(self.handler)
        self.handler.close()
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
        #logger.log(0,msg)
        self.log_level.setValue(selected)
        self.set_level(selected)

        #self.edi_txt.setText('Start logging messages in QWLoggerStd') #logger.getLogContent())


    def save_log_in_file(self):
        logger.info('save_log_in_file ' + self.log_file.value())
        path = str(QFileDialog.getSaveFileName(self,
                                               caption   = 'Select the file to save log',
                                               directory = self.log_file.value(),
                                               filter    = '*.txt'
                                               ))
        if path == '' :
            logger.debug('Saving is cancelled.')
            return 
        self.log_file.setValue(path)
        logger.info('TBD ????????????????  Output file: ' + path)

        #logger.save_log_in_file(path)


    #def save_log_total_in_file(self):
    #    logger.info('save_log_total_in_file' + self.fname_log_total, self._name)
    #    logger.save_log_total_in_file(self.fname_log_total)


    def append_qwlogger(self, msg='...'):
        self.edi_txt.append(msg)
        self.scrollDown()


    def scrollDown(self):
        #logger.debug('scrollDown')
        #scrol_bar_v = self.edi_txt.verticalScrollBar() # QScrollBar
        #scrol_bar_v.setValue(scrol_bar_v.maximum()) 
        self.edi_txt.moveCursor(QTextCursor.End)
        self.edi_txt.repaint()
        #self.raise_()
        #self.edi_txt.update()

#------------------------------

if __name__ == "__main__" :
    import sys
    from psana.pyalgos.generic.PSConfigParameters import PSConfigParameters
    from PyQt5.QtWidgets import QApplication

    #from psana.pyalgos.generic.Logger import logger as log
    #logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    cp = PSConfigParameters()

    app = QApplication(sys.argv)
    w = QWLoggerStd(cp)
    w.setWindowTitle(w._name)
    w.setGeometry(200, 400, 600, 300)

    from psana.graphqt.QWIcons import icon # should be imported after QApplication
    icon.set_icons()
    w.setWindowIcon(icon.icon_logviewer)

    w.show()
    app.exec_()
    sys.exit(0)

#------------------------------
