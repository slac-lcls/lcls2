#------------------------------
"""
:py:class:`QWLoggerStd` - GUI for python logger
===============================================

Usage::
    # Test: python lcls2/psdaq/psdaq/control_gui/QWLoggerStd.py

    # Import
    from psdaq.control_gui.QWLoggerStd import QWLoggerStd

    # Methods - see test

See:
    - :py:class:`QWLoggerStd`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Copied and modified from lcls2/psana/psana/grephqt/QWLoggerStd.py on 2019-01-28 by Mikhail Dubrovin
"""
#------------------------------

import logging

from psalg.utils.syslog import SysLog

logger = logging.getLogger() # need in root to intercept messages from all other loggers
#logger = logging.getLogger(__name__)

from random import randint

from PyQt5.QtWidgets import QWidget, QTextEdit, QLabel, QPushButton, QComboBox,\
                            QHBoxLayout, QVBoxLayout, QFileDialog, QSizePolicy, QSplitter
from PyQt5.QtGui import QTextCursor, QColor
from PyQt5.QtCore import Qt, QSize
from psdaq.control_gui.Styles import style

import psdaq.control_gui.Utils as gu
from psdaq.control_gui.QWLoggerError import QWLoggerError

#------------------------------

# 'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG','NOTSET'

MSG_LEVEL_TO_BKGD_COLOR = {'<C>' : Qt.black, 
                           '<E>' : Qt.yellow,
                           '<W>' : Qt.green,
                           '<I>' : Qt.white,
                           '<D>' : QColor(230, 230, 255),
                           '<N>' : QColor(255, 0, 255)}

MSG_LEVEL_TO_TEXT_COLOR = {'<C>' : Qt.gray, 
                           '<E>' : Qt.red,
                           '<W>' : Qt.magenta,
                           '<I>' : Qt.black,
                           '<D>' : QColor(0, 0, 255),
                           '<N>' : QColor(255, 0, 255)}

#------------------------------

def is_error_msg(rec) :
    return rec.levelname == '<E>'


def log_file_name(lfpath='.') :
    """Returns (str) log file name like /reg/g/psdm/logs/calibman/lcls2/20180518T122407-dubrovin.txt
    """
    if lfpath in (None,'') : return None

    from time import time, strftime, localtime
    from getpass import getuser
    t0_sec = time()
    t0_localtime = localtime(t0_sec)
    tstamp = strftime('%Y%m%dT%H%M%S', t0_localtime)
    #year = strftime('%Y', t0_localtime)
    return '%s/%s-%s.txt' % (lfpath, tstamp, getuser())#, os.getpid())

#------------------------------

class QWFilter(logging.Filter) :
    def __init__(self, qwlogger) :
        #logging.Filter.__init__(self)#, name='')
        self.qwl = qwlogger


    def filter(self, rec) :
        msg = self.qwl.formatter.format(rec)
        self.qwl.set_msg_style(rec.levelname)
        self.qwl.append_qwlogger(msg)
        if is_error_msg(rec) : self.qwl.append_qwlogger_err(msg)
        #self.print_filter_attributes(rec)
        return True


    def print_filter_attributes(self, rec) :
        print('type(rec): %s'%type(rec))
        print('dir(rec): %s'%dir(rec))
        #print('dir(logger): %s'%dir(logger))
        #print('dir(syslog): %s'%dir(self.syslog))
        print(rec.created, rec.name, rec.levelname, rec.msg)

#------------------------------

class QWLoggerStd(QWidget) :
    _name = 'QWLoggerStd'

    def __init__(self, **kwargs) :

        QWidget.__init__(self, parent=None)

        self.log_level    = kwargs.get('log_level', 'DEBUG')
        self.show_buttons = kwargs.get('show_buttons', True)
        self.instrument   = kwargs.get('instrument', 'NonDefined')
        self.log_fname    = log_file_name(kwargs.get('log_prefix', '.'))

        if self.instrument is None: self.instrument='None'

        if self.log_level=='DEBUG' :
            print('%s.__init__ log_fname: %s' % (self._name, self.log_fname))

        if self.log_fname is not None :
            #depth = 6 if self.log_fname[0]=='/' else 1
            gu.create_path(self.log_fname, mode=0o0777)
            #print('Log file: %s' % log_fname)

        #cp.qwloggerstd = self

        #logger.debug('logging.DEBUG: ', logging.DEBUG)
        logger.debug('logging._levelToName: ', logging._levelToName) # {0: 'NOTSET', 50: 'CRITICAL', 20: 'INFO',...
        logger.debug('logging._nameToLevel: ', logging._nameToLevel) # {'NOTSET': 0, 'ERROR': 40, 'WARNING': 30,...

        self.dict_level_to_name = logging._levelToName
        self.dict_name_to_level = logging._nameToLevel
        self.level_names = list(logging._levelToName.values())
        
        self.edi_txt   = QTextEdit('ALL messages')
        #self.edi_err   = QTextEdit('Error messages')
        self.edi_err   = QWLoggerError()
        self.lab_level = QLabel('Log level:')
        self.but_close = QPushButton('&Close') 
        self.but_save  = QPushButton('&Save log-file') 
        self.but_rand  = QPushButton('&Random') 
        self.cmb_level = QComboBox(self) 
        self.cmb_level.addItems(self.level_names)
        self.cmb_level.setCurrentIndex(self.level_names.index(self.log_level))
        
        self.hboxB = QHBoxLayout()
        self.hboxB.addStretch(4)     
        self.hboxB.addWidget(self.lab_level)
        self.hboxB.addWidget(self.cmb_level)
        self.hboxB.addWidget(self.but_rand)
        self.hboxB.addStretch(1)     
        self.hboxB.addWidget(self.but_save)
        self.hboxB.addWidget(self.but_close)

        self.vspl = QSplitter(Qt.Vertical)
        self.vspl.addWidget(self.edi_txt)
        self.vspl.addWidget(self.edi_err)

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.vspl)
        self.vbox.addLayout(self.hboxB)
        self.setLayout(self.vbox)

        if self.show_buttons : self.connect_buttons()

        self.set_style()
        self.set_tool_tips()

        self.config_logger()


    def config_logger(self) :
        
        levname = self.log_level
        level = self.dict_name_to_level.get(levname, logging.DEBUG)

        self.syslog = SysLog(instrument=self.instrument, level=level)

        for h in logger.handlers :
            print('XXX handler:', str(h))

        tsfmt='%Y-%m-%dT%H:%M:%S'
        fmt = '%(levelname)s %(asctime)s %(name)s: %(message)s' if level==logging.DEBUG else\
              '%(asctime)s %(levelname)s: %(message)s'
              #'%(asctime)s %(levelname)s %(name)s: %(message)s'

        #sys.stdout = sys.stderr = open('/dev/null', 'w')

        self.formatter = logging.Formatter(fmt, datefmt=tsfmt)

        self._myfilter = QWFilter(self)
        #self.syslog.syslog_handler.addFilter(self._myfilter)
        self.syslog.console_handler.addFilter(self._myfilter)



    def config_logger_v0(self, log_fname='control_gui.txt') :

        self.append_qwlogger('Start logger\nLog file: %s' % log_fname)

        levname = self.log_level
        level = self.dict_name_to_level.get(levname, logging.DEBUG) # e.g. logging.DEBUG

        #print('YYYYY config_logger levname:', levname, ' level:', level, ' logging.DEBUG:', logging.DEBUG)

        tsfmt='%Y-%m-%dT%H:%M:%S'
        fmt = '%(levelname)s %(name)s: %(message)s' if level==logging.DEBUG else\
              '%(asctime)s %(levelname)s: %(message)s'
              #'%(asctime)s %(levelname)s %(name)s: %(message)s'

        #sys.stdout = sys.stderr = open('/dev/null', 'w')

        self.formatter = logging.Formatter(fmt, datefmt=tsfmt)
        #logger.addFilter(QWFilter(self)) # register self for callback from filter
        # TRICK: add filter to handler to intercept ALL messages

        fname = log_fname if log_fname is not None else '/var/tmp/control_gui_%s.log' % gu.get_login()
        do_save_log_file = (log_fname is not None) or (level==logging.DEBUG)

        self.handler = logging.FileHandler(fname, 'w') if do_save_log_file else\
                       logging.StreamHandler()
        #self.handler = logging.FileHandler(fname, 'w')
        #self.handler = logging.StreamHandler()

        self._myfilter = QWFilter(self)
        self.handler.addFilter(self._myfilter)
        #self.handler.setLevel(logging.NOTSET) # level
        self.handler.setFormatter(self.formatter)
        logger.addHandler(self.handler)
        self.set_level(levname) # pass level name
        print('logging.FileHandler file: %s' % fname)


    def set_level(self, level_name='DEBUG') :
        #self.append_qwlogger('Set logger layer: %s' % level_name)
        #logger.setLevel(level_name) # {0: 'NOTSET'}
        level = self.dict_name_to_level.get(level_name, logging.DEBUG)
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
        self.edi_txt    .setToolTip('Window for ALL messages')
        self.edi_err    .setToolTip('Window for ERROR messages')
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
        #self.edi_err   .setReadOnly(True)
        #self.edi_err   .setStyleSheet(style.styleYellowish) 
        #self.edi_txt   .ensureCursorVisible()
        #self.lab_title.setAlignment(QtCore.Qt.AlignCenter)
        #self.titTitle.setBold()

        self.lab_level .setVisible(self.show_buttons)
        self.cmb_level .setVisible(self.show_buttons)
        self.but_save  .setVisible(self.show_buttons)
        self.but_rand  .setVisible(self.show_buttons)
        self.but_close .setVisible(self.show_buttons)

        #if not self.show_buttons : 
        self.layout().setContentsMargins(2,2,2,2)

        self.edi_err.setMinimumHeight(50)
        self.edi_err.setTextColor(MSG_LEVEL_TO_TEXT_COLOR['<E>'])

        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)

        edi_err_hight = 100
        self.vspl.setSizes((self.vspl.height()-edi_err_hight, edi_err_hight,))

#--------------------

    def sizeHint(self):
        return QSize(300,300)

#--------------------

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
        logger.info('%s.closeEvent' % self._name)
        #self.save_log_total_in_file() # It will be saved at closing of GUIMain
        #self.syslog.syslog_handler.removeFilter(self._myfilter)
        self.syslog.console_handler.removeFilter(self._myfilter)

        #self.handler.removeFilter(self._myfilter)
        #self.handler.close()
        QWidget.closeEvent(self, e)

        #logging.shutdown()
        #print('Exit QWLoggerStd.closeEvent')


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
        self.log_level = selected
        self.set_level(selected)

        #self.edi_txt.setText('Start logging messages in QWLoggerStd') #logger.getLogContent())


    def save_log_in_file(self):
        logger.info('save_log_in_file ' + self.log_fname)
        resp = QFileDialog.getSaveFileName(self,
                                           caption   = 'Select the file to save log',
                                           directory = self.log_fname,
                                           filter    = '*.txt'
                                           )
        logger.debug('save_log_in_file resp: %s' % str(resp))
        path, ftype = resp
        if path == '' :
            logger.debug('Saving is cancelled.')
            return 
        self.log_fname = path
        logger.info('Output file: ' + path)

        #logger.save_log_in_file(path)


    #def save_log_total_in_file(self):
    #    logger.info('save_log_total_in_file' + self.fname_log_total, self._name)
    #    logger.save_log_total_in_file(self.fname_log_total)

    def set_msg_style(self, levelname):
        #print('QWLoggerStd.set_msg_style for level %s' % levelname)
        textcolor = MSG_LEVEL_TO_TEXT_COLOR.get(levelname, Qt.magenta)
        self.edi_txt.setTextColor(textcolor)
        #self.edi_txt.setTextBackgroundColor(bkgdcolor)
        #bkgdcolor = MSG_LEVEL_TO_BKGD_COLOR.get(levelname, Qt.magenta)


    def append_qwlogger(self, msg='...'):
        self.edi_txt.append(msg)
        self.scroll_down_txt()


    def add_separator(self, sep='\n\n\n\n\n%s'%(50*'_')):
        self.append_qwlogger(msg=sep)


    def append_qwlogger_err(self, msg='...'):
        self.edi_err.append(msg)
        self.edi_err.scroll_down()


    def add_separator_err(self, sep='\n\n\n\n\n%s'%(50*'_')):
        self.append_qwlogger_err(msg=sep)


    def scroll_down_txt(self):
        #logger.debug('scroll_down_txt')
        self.edi_txt.moveCursor(QTextCursor.End)
        self.edi_txt.repaint()

    if __name__ == "__main__" :

      def key_usage(self) :
        return 'Keys:'\
               '\n  ESC - exit'\
               '\n  A - add separator in main logger window'\
               '\n  S - add separator in error logger window'\
               '\n'


      def keyPressEvent(self, e) :
        #logger.info('keyPressEvent, key=', e.key())
        if   e.key() == Qt.Key_Escape :
            self.close()

        elif e.key() == Qt.Key_S : 
            self.add_separator_err()

        elif e.key() == Qt.Key_A : 
            self.add_separator()

        else :
            logger.info(self.key_usage())

#------------------------------

if __name__ == "__main__" :
    import sys

    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    app = QApplication(sys.argv)
    w = QWLoggerStd()
    w.setWindowTitle(w._name)
    w.setGeometry(200, 400, 600, 300)

    from psdaq.control_gui.QWIcons import icon # should be imported after QApplication
    icon.set_icons()
    w.setWindowIcon(icon.icon_logviewer)

    w.show()
    app.exec_()
    sys.exit(0)

#------------------------------
