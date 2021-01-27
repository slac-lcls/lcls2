"""
Class :py:class:`CMWMain` is a QWidget for interactive image
============================================================

Usage ::

    import sys
    from PyQt5.QtWidgets import QApplication
    from psana.graphqt.CMWMain import CMWMain
    app = QApplication(sys.argv)
    w = CMWMain(None, app)
    w.show()
    app.exec_()

See:
    - :class:`CMWMain`
    - :class:`CMWMainTabs`
    - :class:`CMConfigParameters`
    - `graphqt documentation <https://lcls-psana.github.io/graphqt/py-modindex.html>`_.

Created on 2017-02-01 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-26 by Mikhail Dubrovin
"""
import logging
logger = logging.getLogger(__name__)

#---

from math import floor
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QTextEdit
from PyQt5.QtGui import QPen, QBrush
from PyQt5.QtCore import Qt, QPoint

from psana.pyalgos.generic.Utils import print_kwargs, print_parser, is_in_command_line, log_rec_on_start
from psana.graphqt.CMConfigParameters import cp
from psana.graphqt.QWLoggerStd import QWLoggerStd
from psana.graphqt.Frame  import Frame
from psana.graphqt.QWIcons import icon
from psana.graphqt.Styles import style




#from psana.graphqt.CMWMainTabs import CMWMainTabs # AFTER proc_parser !!!!

#---

class CMWMain(QWidget):

    _name = 'CMWMain'

    #def __init__(self, parser=None): # **dict_opts):
    def __init__(self, *args, **opts): # **dict_opts):
        QWidget.__init__(self, parent=None)
        #self._name = self.__class__.__name__

        cp.cmwmain = self

        #self.proc_parser(parser)
        self.proc_opts(**opts)

        from psana.graphqt.CMWMainTabs import CMWMainTabs # AFTER proc_parser !!!!

        self.main_win_width  = cp.main_win_width 
        self.main_win_height = cp.main_win_height
        self.main_win_pos_x  = cp.main_win_pos_x 
        self.main_win_pos_y  = cp.main_win_pos_y  

        #icon.set_icons()

        self.wtab = CMWMainTabs()
        self.wlog = QWLoggerStd(cp, show_buttons=False)
        self.wtmp = QTextEdit('Some text')

        self.vspl = QSplitter(Qt.Vertical)
        self.vspl.addWidget(self.wtab) 
        self.vspl.addWidget(self.wlog) 

        self.mbox = QHBoxLayout() 
        self.mbox.addWidget(self.vspl)
        self.setLayout(self.mbox)

        self.set_style()
        #self.set_tool_tips()

        self.connect_signals_to_slots()
        #self.move(self.pos()) # + QPoint(self.width()+5, 0))
        logger.info(log_rec_on_start('%Y-%m-%dT%H:%M:%S%z'))


    def connect_signals_to_slots(self):
        pass
        #self.connect(self.wbut.but_reset, QtCore.SIGNAL('clicked()'), self.on_but_reset)
        #self.connect(self.wbut.but_save,  QtCore.SIGNAL('clicked()'), self.on_but_save)


    def proc_opts(self, **opts):

        host       = opts.get('host', None) # self.opts['host']
        port       = opts.get('port', None) # self.opts['port']
        cp.user    = opts.get('user', None)
        cp.upwd    = opts.get('upwd', None)
        exp        = opts.get('experiment', None)
        det        = opts.get('detector', None)
        logdir     = opts.get('logdir', None)
        loglevel   = opts.get('loglevel', None).upper()
        if isinstance(loglevel,str): loglevel = loglevel.upper()

        if is_in_command_line(None, '--host')      : cp.cdb_host.setValue(host)
        if is_in_command_line(None, '--port')      : cp.cdb_port.setValue(port)
        if is_in_command_line('-e', '--experiment'): cp.exp_name.setValue(exp)
        if is_in_command_line('-d', '--detector')  : cp.data_source.setValue(det)
        if is_in_command_line('-l', '--loglevel')  : cp.log_level.setValue(loglevel)
        if is_in_command_line('-L', '--logdir')    : cp.log_prefix.setValue(logdir)

        if loglevel == 'DEBUG':
            print(40*'_')
            print_kwargs(opts)
            exit('TEST EXIT')


    def proc_parser(self, parser=None):
        self.parser=parser

        if parser is None:
            return

        (popts, pargs) = parser.parse_args()
        self.args = pargs
        self.opts = vars(popts)
        self.defs = vars(parser.get_default_values())

        host       = popts.host # self.opts['host']
        port       = popts.port # self.opts['port']
        cp.user    = popts.user
        cp.upwd    = popts.upwd
        exp        = popts.experiment
        det        = popts.detector
        loglevel   = popts.loglevel.upper()
        logdir     = popts.logdir

        #if host     != self.defs['host']      : cp.cdb_host.setValue(host)
        #if port     != self.defs['port']      : cp.cdb_port.setValue(port)
        #if exp      != self.defs['experiment']: cp.exp_name.setValue(exp)
        #if det      != self.defs['detector']  : cp.data_source.setValue(det)
        #if loglevel != self.defs['loglevel']  : cp.log_level.setValue(loglevel)
        #if logdir   != self.defs['logdir']    : cp.log_prefix.setValue(logdir)

        if is_in_command_line(None, '--host')      : cp.cdb_host.setValue(host)
        if is_in_command_line(None, '--port')      : cp.cdb_port.setValue(port)
        if is_in_command_line('-e', '--experiment'): cp.exp_name.setValue(exp)
        if is_in_command_line('-d', '--detector')  : cp.data_source.setValue(det)
        if is_in_command_line('-l', '--loglevel')  : cp.log_level.setValue(loglevel)
        if is_in_command_line('-L', '--logdir')    : cp.log_prefix.setValue(logdir)

        if loglevel == 'DEBUG':
            print(40*'_')
            print_parser(parser)
            print_kwargs(self.opts)


    def set_tool_tips(self):
        pass
        #self.butStop.setToolTip('Not implemented yet...')


    def set_style(self):
        #self.setGeometry(50, 50, 500, 600)
        self.setGeometry(self.main_win_pos_x .value(),\
                         self.main_win_pos_y .value(),\
                         self.main_win_width .value(),\
                         self.main_win_height.value())
        w_height = self.main_win_height.value()

        self.setMinimumSize(500, 400)

        w = self.main_win_width.value()

        self.layout().setContentsMargins(0,0,0,0)

        spl_pos = cp.main_vsplitter.value()
        self.vspl.setSizes((spl_pos,w_height-spl_pos,))

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

        #try: self.wspe.close()
        #except: pass

        self.wtab.close()

        self.on_save()

        QWidget.closeEvent(self, e)

 
    def resizeEvent(self, e):
        #logger.debug('resizeEvent', self._name) 
        #logger.info('CMWMain.resizeEvent: %s' % str(self.size()))
        pass


    def moveEvent(self, e):
        #logger.debug('moveEvent', self._name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position), __name__)       
        #logger.info('CMWMain.moveEvent - move window to x,y: ', str(self.mapToGlobal(QPoint(0,0))))
        #self.wimg.move(self.pos() + QPoint(self.width()+5, 0))
        pass


    def key_usage(self):
        return 'Keys:'\
               '\n  V - view/hide tabs'\
               '\n'


    if __name__ == "__main__":
      def keyPressEvent(self, e):
        #print('keyPressEvent, key=', e.key())       
        if   e.key() == Qt.Key_Escape:
            self.close()
        elif e.key() == Qt.Key_V: 
            self.wtab.view_hide_tabs()
        else:
            logger.info(self.key_usage())


    def on_save(self):

        point, size = self.mapToGlobal(QPoint(-5,-22)), self.size() # Offset (-5,-22) for frame size.
        x,y,w,h = point.x(), point.y(), size.width(), size.height()
        msg = 'Save main window x,y,w,h: %d, %d, %d, %d' % (x,y,w,h)
        logger.info(msg) #, self._name)
        #print(msg)

        #Save main window position and size
        self.main_win_pos_x .setValue(x)
        self.main_win_pos_y .setValue(y)
        self.main_win_width .setValue(w)
        self.main_win_height.setValue(h)

        spl_pos = self.vspl.sizes()[0]
        msg = 'Save main v-splitter position %d' % spl_pos
        logger.debug(msg)

        cp.main_vsplitter.setValue(spl_pos)

        cp.printParameters()
        cp.saveParametersInFile() # moved to PSConfigParameters

        if cp.save_log_at_exit.value():
            pass
            # ?????
            #log.saveLogInFile(cp.log_file.value())
            #print('Saved log file: %s' % cp.log_file.value())
            #log.saveLogTotalInFile(fnm.log_file_total())


#def calibman(parser=None):
def calibman(*args,**opts):
    import sys
    #sys.stdout = sys.stderr = open('/dev/null', 'w') # open('%s-stdout-stderr' % cp.log_file.value(), 'w')

    from PyQt5.QtWidgets import QApplication
    #logging.basicConfig(format='[%(levelname).1s] %(asctime)s L:%(lineno)03d %(message)s', datefmt='%Y-%m-%dT%H:%M:%S', level=logging.DEBUG)
    app = QApplication(sys.argv)
    #w = CMWMain(parser)
    w = CMWMain(*args,**opts)
    w.move(0,20)
    w.show()
    app.exec_()
    del w
    del app


if __name__ == "__main__":
    calibman()

# EOF
