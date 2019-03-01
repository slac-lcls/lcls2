"""
Class :py:class:`CGWMain` is a QWidget for interactive image
============================================================

Usage ::

    import sys
    from PyQt5.QtWidgets import QApplication
    from psdaq.control_gui.CGWMain import CGWMain
    app = QApplication(sys.argv)
    w = CGWMain(None, app)
    w.show()
    app.exec_()

See:
    - :class:`CGWMain`
    - :class:`CGWMainPartition`
    - `graphqt documentation <https://lcls-psana.github.io/graphqt/py-modindex.html>`_.

Created on 2019-01-25 by Mikhail Dubrovin
"""
#import os
#import math
#from math import floor

#------------------------------

import logging
logger = logging.getLogger(__name__)

#------------------------------
import json
from time import time

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QTextEdit
from PyQt5.QtCore import Qt#, QPoint

#from PyQt5.QtGui import QPen, QBrush
#from psdaq.control_gui.CGWMainPartition import CGWMainPartition

from psdaq.control_gui.CGWMainConfiguration import CGWMainConfiguration
from psdaq.control_gui.CGWMainPartition     import CGWMainPartition
from psdaq.control_gui.CGWMainControl       import CGWMainControl
from psdaq.control_gui.CGWMainDetector      import CGWMainDetector
from psdaq.control_gui.CGWMainRunStatistics import CGWMainRunStatistics
from psdaq.control_gui.QWLoggerStd          import QWLoggerStd
from psdaq.control_gui.CGDaqControl         import daq_control, DaqControl

from psdaq.control_gui.QWZMQListener        import QWZMQListener, zmq

#------------------------------

class CGWMain(QWZMQListener) :

    _name = 'CGWMain'

    def __init__(self, parser=None) : # **dict_opts) :

        self.proc_parser(parser)
        daq_control.set_daq_control(DaqControl(host=self.host, platform=self.platform, timeout=self.timeout))

        self.wlogr = QWLoggerStd(log_level=self.loglevel, show_buttons=False, log_prefix=self.logdir)

        QWZMQListener.__init__(self, host=self.host, platform=self.platform, timeout=self.timeout)
 

        #QWidget.__init__(self, parent=None)
        #self._name = self.__class__.__name__

        #cp.cmwmain = self

        #self.main_win_width  = cp.main_win_width 
        #self.main_win_height = cp.main_win_height
        #self.main_win_pos_x  = cp.main_win_pos_x 
        #self.main_win_pos_y  = cp.main_win_pos_y  

        #icon.set_icons()

        #from psana.graphqt.CGWMainTabs import CGWMainTabs

        #self.wtab = CGWMainTabs()
        #self.wlog = QWLogger(log, cp, show_buttons=False)
        #self.wlog = QWLoggerStd(cp, show_buttons=False)
        self.wconf = CGWMainConfiguration()
        self.wpart = CGWMainPartition()
        self.wctrl = CGWMainControl(parent_ctrl=self)
        self.wdetr = CGWMainDetector(parent_ctrl=self)
        self.wrsta = CGWMainRunStatistics()
        #self.wlogr = QTextEdit('my logger')

        #self.vbox = QVBoxLayout() 
        #self.vbox.addWidget(self.wtab) 
        #self.vbox.addStretch(1)

        #self.wrig = QWidget()
        #self.wrig.setLayout(self.vbox)

        self.vspl = QSplitter(Qt.Vertical)
        self.vspl.addWidget(self.wconf) 
        self.vspl.addWidget(self.wpart) 
        self.vspl.addWidget(self.wctrl) 
        self.vspl.addWidget(self.wdetr) 
        self.vspl.addWidget(self.wrsta) 
        self.vspl.addWidget(self.wlogr) 

        #self.hspl = QSplitter(Qt.Horizontal)
        #self.hspl.addWidget(self.vspl)
        #self.hspl.addWidget(self.wtmp)
        #self.hspl.addWidget(self.wrig)

        self.mbox = QHBoxLayout() 
        self.mbox.addWidget(self.vspl)
        self.setLayout(self.mbox)

        self.set_style()
        #self.set_tool_tips()

        #self.connect_signals_to_slots()

        #self.move(self.pos()) # + QPoint(self.width()+5, 0))
        self.setWindowTitle("DAQ Control")

    def connect_signals_to_slots(self) :
        pass
        #self.connect(self.wbut.but_reset, QtCore.SIGNAL('clicked()'), self.on_but_reset)
        #self.connect(self.wbut.but_save,  QtCore.SIGNAL('clicked()'), self.on_but_save)

#------------------------------

    def proc_parser(self, parser=None) :
        self.parser=parser

        if parser is None :
            return

        (popts, pargs) = parser.parse_args()
        self.args = pargs
        self.opts = vars(popts)
        self.defs = vars(parser.get_default_values())

        #host       = popts.host     # self.opts['host']
        #port       = popts.port # self.opts['port']
        #cp.user    = popts.user
        #cp.upwd    = popts.upwd
        #exp        = popts.experiment
        #det        = popts.detector
        self.logdir   = popts.logdir
        self.loglevel = popts.loglevel.upper()
        self.host     = popts.host
        self.platform = popts.platform
        self.timeout  = popts.timeout

        #if host     != self.defs['host']       : cp.cdb_host.setValue(host)
        #if host     != self.defs['host']       : cp.cdb_host.setValue(host)
        #if port     != self.defs['port']       : cp.cdb_port.setValue(port)
        #if exp      != self.defs['experiment'] : cp.exp_name.setValue(exp)
        #if det      != self.defs['detector']   : cp.data_source.setValue(det)
        #if loglevel != self.defs['loglevel']   : cp.log_level.setValue(loglevel)
        #if logdir   != self.defs['logdir']     : cp.log_prefix.setValue(logdir)

        #if is_in_command_line(None, '--host')       : cp.cdb_host.setValue(host)
        #if is_in_command_line(None, '--port')       : cp.cdb_port.setValue(port)
        #if is_in_command_line('-e', '--experiment') : cp.exp_name.setValue(exp)
        #if is_in_command_line('-d', '--detector')   : cp.data_source.setValue(det)
        #if is_in_command_line('-l', '--loglevel')   : cp.log_level.setValue(loglevel)
        #if is_in_command_line('-L', '--logdir')     : cp.log_prefix.setValue(logdir)

        #if self.loglevel == 'DEBUG' :
        #    print(40*'_')
        #    print_parser(parser)
        #    print_kwargs(self.opts)

#------------------------------

    def set_tool_tips(self) :
        pass
        #self.butStop.setToolTip('Not implemented yet...')


    def set_style(self) :

        self.setMinimumWidth(350)

        #self.setGeometry(50, 50, 500, 600)
        #self.setGeometry(self.main_win_pos_x .value(),\
        #                 self.main_win_pos_y .value(),\
        #                 self.main_win_width .value(),\
        #                 self.main_win_height.value())
        #w_height = self.main_win_height.value()

        #self.setMinimumSize(500, 400)

        #self.layout().setContentsMargins(0,0,0,0)

        from psdaq.control_gui.QWIcons import icon
        icon.set_icons()
        self.setWindowIcon(icon.icon_button_ok)

        #w = self.main_win_width.value()
        #spl_pos = cp.main_vsplitter.value()
        #self.vspl.setSizes((spl_pos,w_height-spl_pos,))

        #self.wrig.setMinimumWidth(350)
        #self.wrig.setMaximumWidth(450)

        #self.wrig.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        #self.hspl.moveSplitter(w*0.5,0)

        #self.setFixedSize(800,500)
        #self.setMinimumSize(500,800)

        #self.setStyleSheet("background-color:blue; border: 0px solid green")
        #self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        
        #self.        setStyleSheet(style.styleBkgd)
        #self.butSave.setStyleSheet(style.styleButton)
        #self.butExit.setStyleSheet(style.styleButton)
        #self.butELog.setStyleSheet(style.styleButton)
        #self.butFile.setStyleSheet(style.styleButton)

        #self.butELog    .setVisible(False)
        #self.butFBrowser.setVisible(False)

        #self.but1.raise_()


    def closeEvent(self, e) :
        logger.debug('%s.closeEvent' % self._name)

        #try : self.wspe.close()
        #except : pass

        #self.wtab.close()

        #self.on_save()

        QWidget.closeEvent(self, e)

 
    def resizeEvent(self, e):
        #logger.debug('resizeEvent', self._name) 
        #logger.info('CGWMain.resizeEvent: %s' % str(self.size()))
        pass


    def moveEvent(self, e) :
        #logger.debug('moveEvent', self._name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position), __name__)       
        #logger.info('CGWMain.moveEvent - move window to x,y: ', str(self.mapToGlobal(QPoint(0,0))))
        #self.wimg.move(self.pos() + QPoint(self.width()+5, 0))
        pass


 
    def on_save(self):

        point, size = self.mapToGlobal(QPoint(-5,-22)), self.size() # Offset (-5,-22) for frame size.
        x,y,w,h = point.x(), point.y(), size.width(), size.height()
        msg = 'Save main window x,y,w,h : %d, %d, %d, %d' % (x,y,w,h)
        logger.info(msg) #, self._name)
        #print(msg)

        #Save main window position and size
        #self.main_win_pos_x .setValue(x)
        #self.main_win_pos_y .setValue(y)
        #self.main_win_width .setValue(w)
        #self.main_win_height.setValue(h)

        spl_pos = self.vspl.sizes()[0]
        msg = 'Save main v-splitter position %d' % spl_pos
        logger.debug(msg)

        #cp.main_vsplitter.setValue(spl_pos)

        #cp.printParameters()
        #cp.saveParametersInFile() # moved to PSConfigParameters

        #if cp.save_log_at_exit.value() :
        #    pass
            # ?????
            #log.saveLogInFile(cp.log_file.value())
            #print('Saved log file: %s' % cp.log_file.value())
            #log.saveLogTotalInFile(fnm.log_file_total())


#------------------------------

    def on_zmq_poll(self):
        """Re-implementation of the superclass QWZMQListener method for zmq message processing.
        """
        t0_sec = time()

        self.zmq_notifier.setEnabled(False)

        flags = self.zmq_socket.getsockopt(zmq.EVENTS)

        flag = 'UNKNOWN'
        msg = ''
        if flags & zmq.POLLIN :
          while self.zmq_socket.getsockopt(zmq.EVENTS) & zmq.POLLIN :
            flag = 'POLLIN'
            msg = self.zmq_socket.recv_multipart()
            self.process_zmq_message(msg)
            #self.setWindowTitle(str(msg))
        elif flags & zmq.POLLOUT : flag = 'POLLOUT'
        elif flags & zmq.POLLERR : flag = 'POLLERR'
        else : pass

        self.zmq_notifier.setEnabled(True)
        _ = self.zmq_socket.getsockopt(zmq.EVENTS) # WITHOUT THIS LINE IT WOULD NOT CALL on_read_msg AGAIN!
        logger.debug('CGWMain.on_zmq_poll Flag zmq.%s in %d msg: %s' % (flag, flags, msg)\
                   + '\n    poll processing time = %.6f sec' % (time()-t0_sec))


    def process_zmq_message(self, msg):
        #print('==== msg: %s' % str(msg))
        try :
            for rec in msg :
                ucode = rec.decode('utf8').replace("\'t", ' not').replace("'", '"')
                jo = json.loads(ucode)
                sj = json.dumps(jo, indent=2, sort_keys=False)
                #logger.debug("msg as json:\n%s" % sj)
                #  jo['header'] # {'key': 'status', 'msg_id': '0918505109-317821000', 'sender_id': None}
                #  jo['body']   # {'state': 'allocated', 'transition': 'alloc'}

                if  jo['header']['key'] == 'status' :
                    s_state      = jo['body']['state']
                    s_transition = jo['body']['transition']
                    self.wdetr.set_but_state (s_state)
                    self.wctrl.set_transition(s_transition)
                    logging.info('received state msg: %s and transition: %s' % (s_state, s_transition))

                elif jo['header']['key'] == 'error' :
                    logging.error('received error msg: %s' % jo['body']['error'])

        except KeyError as ex:
             logger.warning('CGWMain.process_zmq_message: %s\nError: %s' % (str(msg),ex))

        except Exception as ex:
             logger.warning('CGWMain.process_zmq_message: %s\nError: %s' % (str(msg),ex.message))

#------------------------------
#------------------------------
#------------------------------
#------------------------------

    if __name__ == "__main__" :

      def key_usage(self) :
        return 'Keys:'\
               '\n  V - view/hide tabs'\
               '\n'

      def keyPressEvent(self, e) :
        #print('keyPressEvent, key=', e.key())       
        if   e.key() == Qt.Key_Escape :
            self.close()
        elif e.key() == Qt.Key_V : 
            self.wtab.view_hide_tabs()
        else :
            logger.info(self.key_usage())


#------------------------------

def proc_control_gui(parser=None) :
    import sys
    #sys.stdout = sys.stderr = open('/dev/null', 'w') # open('%s-stdout-stderr' % cp.log_file.value(), 'w')

    from PyQt5.QtWidgets import QApplication
    #logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)
    app = QApplication(sys.argv)

    w = CGWMain(parser)

    #print('In CGWMain:proc_control_gui A')
    w.show()
    print('In CGWMain:proc_control_gui after w.show() - ERRORS FROM libGL IS A KNOWN ISSUE')

    app.exec_()
    del w
    del app

#------------------------------

if __name__ == "__main__" :
    proc_control_gui()

#------------------------------
