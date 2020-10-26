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
    - `lcls2 on github <https://github.com/slac-lcls/lcls2/psdaq/psdaq/control_gui>`_.

Created on 2019-01-25 by Mikhail Dubrovin
"""
#------------------------------

import logging
logger = logging.getLogger(__name__)

#------------------------------
# zmq.utils.jsonapi ensures bytes, instead of unicode
import zmq.utils.jsonapi as json
from time import time

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QTextEdit, QSizePolicy
from PyQt5.QtCore import Qt, QSize, QPoint

from psdaq.control_gui.CGConfigParameters   import cp
from psdaq.control_gui.CGWMainConfiguration import CGWMainConfiguration
from psdaq.control_gui.CGWMainInfo          import CGWMainInfo
from psdaq.control_gui.QWLoggerStd          import QWLoggerStd
from psdaq.control_gui.CGDaqControl         import daq_control, DaqControl,\
                                                   daq_control_get_status, daq_control_get_instrument
from psdaq.control_gui.QWZMQListener        import QWZMQListener, zmq
from psdaq.control_gui.QWUtils              import confirm_or_cancel_dialog_box
#from psdaq.control_gui.CGWMainTabs          import CGWMainTabs
from psdaq.control_gui.CGWMainTabExpert     import CGWMainTabExpert

#------------------------------

class CGWMain(QWZMQListener):

    _name = 'CGWMain'

    def __init__(self, parser=None):

        self.proc_parser(parser)

        if __name__ != "__main__":
          daq_control.set_daq_control(DaqControl(host=self.host, platform=self.platform, timeout=self.timeout))
          QWZMQListener.__init__(self, host=self.host, platform=self.platform, timeout=self.timeout)
        else: # emulator mode for TEST ONLY
          QWZMQListener.__init__(self, is_normal=False)

        self.init_daq_control_parameters() # cach parameters in cp

        self.wlogr = QWLoggerStd(log_level=self.loglevel, instrument=cp.instr,\
                                 log_prefix=self.logdir, show_buttons=False)
 
        logger.debug('logger started with log_level:%s instrument:%s' % (self.loglevel, cp.instr))

        cp.cgwmain = self

        self.main_win_width  = cp.main_win_width 
        self.main_win_height = cp.main_win_height
        self.main_win_pos_x  = cp.main_win_pos_x 
        self.main_win_pos_y  = cp.main_win_pos_y  

        #icon.set_icons()

        self.wconf = CGWMainConfiguration()
        #self.wtabs = CGWMainTabs()
        self.wtabs = CGWMainTabExpert()
        self.winfo = CGWMainInfo()

        self.vspl = QSplitter(Qt.Vertical)
        self.vspl.addWidget(self.wconf)
        self.vspl.addWidget(self.wtabs)
        self.vspl.addWidget(self.winfo)
        self.vspl.addWidget(self.wlogr)

        self.mbox = QHBoxLayout() 
        self.mbox.addWidget(self.vspl)
        self.setLayout(self.mbox)

        self.set_style()
        #self.set_tool_tips()
        #self.connect_signals_to_slots()
        #self.move(self.pos()) # + QPoint(self.width()+5, 0))
        self.set_buts_enabled()


    def connect_signals_to_slots(self):
        pass
        #self.connect(self.wbut.but_reset, QtCore.SIGNAL('clicked()'), self.on_but_reset)
        #self.connect(self.wbut.but_save,  QtCore.SIGNAL('clicked()'), self.on_but_save)

#------------------------------

    def proc_parser(self, parser=None):
        self.parser=parser

        if parser is None:
            self.loglevel = 'DEBUG'
            self.logdir   = 'logdir'
            self.expert   = None
            return

        (popts, pargs) = parser.parse_args()
        self.args = pargs
        self.opts = vars(popts)
        self.defs = vars(parser.get_default_values())

        self.logdir     = popts.logdir
        self.loglevel   = popts.loglevel.upper()
        self.host       = popts.host
        self.platform   = popts.platform
        self.timeout    = popts.timeout
        self.expname    = popts.expname
        self.uris       = popts.uris  # 'https://pswww.slac.stanford.edu/ws-auth/devconfigdb/ws/'
        self.expert     = popts.expert # bool
        self.user       = popts.user 
        self.password   = popts.password 

        #if host     != self.defs['host']      : cp.cdb_host.setValue(host)
        #if host     != self.defs['host']      : cp.cdb_host.setValue(host)
        #if port     != self.defs['port']      : cp.cdb_port.setValue(port)
        #if exp      != self.defs['experiment']: cp.exp_name.setValue(exp)
        #if det      != self.defs['detector']  : cp.data_source.setValue(det)
        #if loglevel != self.defs['loglevel']  : cp.log_level.setValue(loglevel)
        #if logdir   != self.defs['logdir']    : cp.log_prefix.setValue(logdir)

        #if is_in_command_line(None, '--host')      : cp.cdb_host.setValue(host)
        #if is_in_command_line(None, '--port')      : cp.cdb_port.setValue(port)
        #if is_in_command_line('-e', '--experiment'): cp.exp_name.setValue(exp)
        #if is_in_command_line('-d', '--detector')  : cp.data_source.setValue(det)
        #if is_in_command_line('-l', '--loglevel')  : cp.log_level.setValue(loglevel)
        #if is_in_command_line('-L', '--logdir')    : cp.log_prefix.setValue(logdir)

        #if self.loglevel == 'DEBUG':
        #    print(40*'_')
        #    print_parser(parser)
        #    print_kwargs(self.opts)
#------------------------------

    def init_daq_control_parameters(self):
        cp.s_transition, cp.s_state, cp.s_cfgtype, cp.s_recording, _platform, \
            cp.s_bypass_activedet, cp.s_experiment_name, cp.s_run_number, cp.s_last_run_number = daq_control_get_status()
        cp.instr = daq_control_get_instrument()
        logger.debug('daq_control_get_instrument(): %s' % cp.instr)
        if cp.instr is None: logger.warning('instrument is None')

#------------------------------

    def set_tool_tips(self):
        pass
        #self.setToolTip('DAQ control')

#--------------------

    def sizeHint(self):
        """Set default window size
        """
        return QSize(370, 810)

#--------------------

    def set_style(self):
        self.setWindowTitle("DAQ Control")
        #self.layout().setContentsMargins(0,0,0,0)
        self.layout().setContentsMargins(3,0,3,0)
        self.setMinimumSize(300,700)

        self.wconf.setFixedHeight(80)

        self.setGeometry(self.main_win_pos_x .value(),\
                         self.main_win_pos_y .value(),\
                         self.main_win_width .value(),\
                         self.main_win_height.value())
        #w_height = self.main_win_height.value()

        #self.layout().setContentsMargins(0,0,0,0)

        from psdaq.control_gui.QWIcons import icon, QIcon

        icon.set_icons()
        #self.setWindowIcon(icon.icon_button_ok)
        self.setWindowIcon(icon.icon_lcls)

        #self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        #self.setFocusPolicy(Qt.StrongFocus)

        #pmap = icon.icon_lcls.pixmap(QSize(200,100)) # change icon.pixmap size
        #self.setWindowIcon(QIcon(pmap))

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


    def closeEvent(self, e):
        logger.debug('%s.closeEvent' % self._name)

        resp = confirm_or_cancel_dialog_box(parent=None,
                                            text='Close window?',\
                                            title='Confirm or cancel') 
        if not resp: 
            logger.warning('Closing window is cancelled')
            e.ignore()
            return

        logger.debug('Closing window is confirmed')

        # save app-configuration parameters
        #try:
        #    self.on_save()
        #except Exception as ex:
        #    print('Exception: %s' % ex)

        try: 
            self.wtabs.close()
            self.wconf.close()
            self.wlogr.close()
        except Exception as ex:
            print('Exception: %s' % ex)

        QWZMQListener.closeEvent(self, e)

        #print('Exit CGWMain.closeEvent')

        cp.cgwmain = None

#--------------------
        
#    def __del__(self):
#        logger.debug('In CGConfigParameters d-tor')
#        #if self.save_cp_at_exit.value():
#        if True:
#            self.on_save()

#--------------------
 
    #def resizeEvent(self, e):
        #logger.debug('CGWMain.resizeEvent: %s' % str(self.size()))
        #QWZMQListener.resizeEvent(self, e)


    #def moveEvent(self, e):
        #logger.debug('moveEvent', self._name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position), __name__)       
        #logger.info('CGWMain.moveEvent - move window to x,y: ', str(self.mapToGlobal(QPoint(0,0))))
        #self.wimg.move(self.pos() + QPoint(self.width()+5, 0))
        #pass

 
    def on_save(self):

        point, size = self.mapToGlobal(QPoint(-5,-22)), self.size() # Offset (-5,-22) for frame size.
        x,y,w,h = point.x(), point.y(), size.width(), size.height()
        msg = 'Save main window x,y,w,h: %d, %d, %d, %d' % (x,y,w,h)
        logger.info(msg) #, self._name)
        print(msg)

        #Save main window position and size
        self.main_win_pos_x .setValue(x)
        self.main_win_pos_y .setValue(y)
        self.main_win_width .setValue(w)
        self.main_win_height.setValue(h)

        spl_pos = self.vspl.sizes()[0]
        msg = 'Save main v-splitter position %d' % spl_pos
        logger.debug(msg)

        #cp.main_vsplitter.setValue(spl_pos)

        cp.printParameters()
        cp.saveParametersInFile()

        #if cp.save_log_at_exit.value():
        #    pass
            # ?????
            #log.saveLogInFile(cp.log_file.value())
            #print('Saved log file: %s' % cp.log_file.value())
            #log.saveLogTotalInFile(fnm.log_file_total())


    def wcontrol(self):
        return cp.cgwmaincontrol
        #return cp.cgwmaintabuser if cp.cgwmaintabuser is not None else cp.cgwmaincontrol


    def set_buts_enabled(self):
        """set buttons in all widgets enabled/disabled depending on current state.
        """
        w = self.wcontrol()
        if w is not None:
            w.set_buts_enabled()
            w.update_progress_bar(0, is_visible=False)

        w = cp.cgwmainpartition
        if w is not None: w.set_buts_enabled()

        w = cp.cgwmainconfiguration
        if w is not None: w.set_buts_enabled()

        w = cp.cgwmaininfo
        if w is not None: w.update_info()

#------------------------------

    def on_zmq_poll(self):
        """Re-implementation of the superclass QWZMQListener method for zmq message processing.
        """
        t0_sec = time()
        self.zmq_notifier.setEnabled(False)
        flags = self.zmq_socket.getsockopt(zmq.EVENTS)
        flag = 'UNKNOWN'
        msg = ''
        if flags & zmq.POLLIN:
          while self.zmq_socket.getsockopt(zmq.EVENTS) & zmq.POLLIN:
            flag = 'POLLIN'
            msg = self.zmq_socket.recv_multipart()
            self.process_zmq_message(msg)
            #self.setWindowTitle(str(msg))
        elif flags & zmq.POLLOUT: flag = 'POLLOUT'
        elif flags & zmq.POLLERR: flag = 'POLLERR'
        else: pass

        self.zmq_notifier.setEnabled(True)
        _flags = self.zmq_socket.getsockopt(zmq.EVENTS) # WITHOUT THIS LINE IT WOULD NOT CALL on_read_msg AGAIN!
        logger.debug('CGWMain.on_zmq_poll Flag zmq.%s in %d msg: %s' % (flag, flags, msg)\
                   + '\n    poll processing time = %.6f sec' % (time()-t0_sec))
        if _flags & zmq.POLLIN: self.on_zmq_poll()


    def process_zmq_message(self, msg):
        #print('==== msg: %s' % str(msg))

        try:
            for rec in msg:
                jo = json.loads(rec)
                #  jo['header'] # {'key': 'status', 'msg_id': '0918505109-317821000', 'sender_id': None}
                #  jo['body']   # {'state': 'allocated', 'transition': 'alloc'}

                jo_header_key = jo['header']['key']

                if  jo_header_key == 'status':
                    body = jo['body']
                    cp.s_transition = body['transition']
                    cp.s_state      = body['state']
                    cp.s_cfgtype    = body['config_alias'] # BEAM/NO BEAM
                    cp.s_recording  = body['recording']    # True/False
                    cp.s_platform   = body.get('platform', None) # dict
                    cp.s_bypass_activedet = body['bypass_activedet'] # True/False
                    cp.s_experiment_name  = body['experiment_name']  # string
                    cp.s_run_number       = body['run_number']       # int
                    cp.s_last_run_number  = body['last_run_number']  # int

                    #====
                    self.wconf.set_config_type(cp.s_cfgtype)
                    w = cp.cgwmaincollection
                    if w is not None: w.update_table()
                    logger.info('zmq msg transition:%s state:%s config:%s recording:%s'%\
                                (cp.s_transition, cp.s_state, cp.s_cfgtype, cp.s_recording))

                elif jo_header_key == 'error':
                    body = jo['body']
                    logger.error(str(body['err_info']))

                elif jo_header_key == 'warning':
                    body = jo['body']
                    logger.warning(str(body['err_info']))

                elif jo_header_key == 'progress':
                    body = jo['body']
                    logger.debug('progress: %s' % str(body))
                    wctrl = self.wcontrol()
                    if wctrl is not None: 
                        v = 100*body['elapsed'] / body['total']
                        wctrl.update_progress_bar(v, is_visible=True, trans_name=body['transition'])
                        return # DO NOT enable_buttons()

                else:
                    sj = json.dumps(jo, indent=2, sort_keys=False)
                    logger.debug('received jason:\n%s' % sj)

                self.set_buts_enabled()

        except KeyError as ex:
             logger.warning('CGWMain.process_zmq_message: %s\nError: %s' % (str(msg),ex))

        except Exception as ex:
             logger.warning('CGWMain.process_zmq_message: %s\nError: %s' % (str(msg),ex))

#------------------------------
#------------------------------
#------------------------------
#------------------------------

    if __name__ == "__main__":

      def key_usage(self):
        return 'Keys:'\
               '\n  V - view/hide tabs'\
               '\n'

      def keyPressEvent(self, e):
        #print('keyPressEvent, key=', e.key())       
        if   e.key() == Qt.Key_Escape:
            self.close()
        elif e.key() == Qt.Key_V: 
            self.wtab.view_hide_tabs()
        else:
            logger.info(self.key_usage())

#------------------------------

def proc_control_gui(parser=None):
    import sys
    #sys.stdout = sys.stderr = open('/dev/null', 'w') # open('%s-stdout-stderr' % cp.log_file.value(), 'w')

    from PyQt5.QtWidgets import QApplication
    #logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)
    app = QApplication(sys.argv)

    cp.qapplication = app
    w = CGWMain(parser)

    w.show()
    print('In CGWMain:proc_control_gui after w.show() - ERRORS FROM libGL IS A KNOWN ISSUE')

    app.exec_()

    del w
    del app

#------------------------------

if __name__ == "__main__":

    from psdaq.control_gui.CGDaqControl import DaqControlEmulator
    daq_control.set_daq_control(DaqControlEmulator())

    proc_control_gui()
    print('End of CGWMain')

#------------------------------
