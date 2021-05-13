"""
Class :py:class:`CMWMain` is a QWidget for interactive image
============================================================

Usage ::
    see calibman

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

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QSplitter, QTextEdit
from PyQt5.QtCore import Qt, QPoint

from psana.pyalgos.generic.Utils import print_kwargs, is_in_command_line, log_rec_on_start
from psana.graphqt.CMConfigParameters import cp
from psana.graphqt.QWLoggerStd import QWLoggerStd


class CMWMain(QWidget):

    _name = 'CMWMain'

    def __init__(self, **kwargs):
        QWidget.__init__(self, parent=None)
        #self._name = self.__class__.__name__

        cp.cmwmain = self

        self.set_input_pars(**kwargs)

        from psana.graphqt.CMWMainTabs import CMWMainTabs # AFTER set_input_pars !!!!\
        self.wlog = QWLoggerStd(cp, show_buttons=False)
        self.wtab = CMWMainTabs() #self.wtab = QTextEdit('Some text')

        self.vspl = QSplitter(Qt.Vertical)
        self.vspl.addWidget(self.wtab) 
        self.vspl.addWidget(self.wlog) 

        self.mbox = QHBoxLayout() 
        self.mbox.addWidget(self.vspl)

        x,y,w,h = self.xywh
        self.setGeometry(x,y,w,h)
        logger.info('set preserved window geometry x,y,w,h: %d,%d,%d,%d' % (x,y,w,h))
        logger.info(log_rec_on_start()) #tsfmt='%Y-%m-%dT%H:%M:%S%z'))

        self.setLayout(self.mbox)

        self.set_style()
        #self.set_tool_tips()
        #self.connect_signals_to_slots()


#    def connect_signals_to_slots(self):
#        pass
#        #self.connect(self.wbut.but_reset, QtCore.SIGNAL('clicked()'), self.on_but_reset)
#        #self.connect(self.wbut.but_save,  QtCore.SIGNAL('clicked()'), self.on_but_save)


    def set_input_pars(self, **kwa):

        cp.kwargs = kwa

        self.xywh = cp.main_win_pos_x .value(),\
                    cp.main_win_pos_y .value(),\
                    cp.main_win_width .value(),\
                    cp.main_win_height.value()

        self.main_win_pos_x  = cp.main_win_pos_x
        self.main_win_pos_y  = cp.main_win_pos_y
        self.main_win_width  = cp.main_win_width
        self.main_win_height = cp.main_win_height

        host       = kwa.get('host', None) # self.kwa['host']
        port       = kwa.get('port', None) # self.kwa['port']
        cp.user    = kwa.get('user', None)
        cp.upwd    = kwa.get('upwd', None)
        exp        = kwa.get('experiment', None)
        det        = kwa.get('detector', None)
        logdir     = kwa.get('logdir', None)
        loglevel   = kwa.get('loglevel', 'DEBUG').upper()
        if isinstance(loglevel,str): loglevel = loglevel.upper()

        if is_in_command_line(None, '--host')      : cp.cdb_host.setValue(host)
        if is_in_command_line(None, '--port')      : cp.cdb_port.setValue(port)
        if is_in_command_line('-e', '--experiment'): cp.exp_name.setValue(exp)
        if is_in_command_line('-d', '--detector')  : cp.data_source.setValue(det)
        if is_in_command_line('-l', '--loglevel')  : cp.log_level.setValue(loglevel)
        if is_in_command_line('-L', '--logdir')    : cp.log_prefix.setValue(logdir)

        if loglevel == 'DEBUG':
            print(40*'_')
            print_kwargs(kwa)


    def set_tool_tips(self):
        self.setToolTip('Calibration Management GUI')


    def set_style(self):
        self.setMinimumSize(500, 400)
        self.layout().setContentsMargins(0,0,0,0)
        spl_pos = cp.main_vsplitter.value()
        w_height = self.main_win_height.value()
        self.vspl.setSizes((spl_pos, w_height-spl_pos,))


    def closeEvent(self, e):
        logger.debug('%s.closeEvent' % self._name)
        #try: self.wspe.close()
        #except: pass
        self.wtab.close()
        self.on_save()
        QWidget.closeEvent(self, e)

 
#    def resizeEvent(self, e):
#        QWidget.resizeEvent(self, e)
#        print('XXX resizeEvent _name', self._name) 


#    def moveEvent(self, e):
#        QWidget.moveEvent(self, e)
#        pass
        #logger.debug('moveEvent', self._name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position), __name__)       
        #logger.info('CMWMain.moveEvent - move window to x,y: ', str(self.mapToGlobal(QPoint(0,0))))
        #self.wimg.move(self.pos() + QPoint(self.width()+5, 0))


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
        msg = 'save window geometry x,y,w,h: %d,%d,%d,%d' % (x,y,w,h)
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

        cp.main_vsplitter.setValue(spl_pos)

        cp.printParameters()
        cp.saveParametersInFile() # moved to PSConfigParameters

        if cp.save_log_at_exit.value():
            pass
            # ?????
            #log.saveLogInFile(cp.log_file.value())
            #print('Saved log file: %s' % cp.log_file.value())
            #log.saveLogTotalInFile(fnm.log_file_total())


def calibman(**kwargs):
    import sys
    #sys.stdout = sys.stderr = open('/dev/null', 'w') # open('%s-stdout-stderr' % cp.log_file.value(), 'w')
    #logging.basicConfig(format='[%(levelname).1s] %(asctime)s L:%(lineno)03d %(message)s', datefmt='%Y-%m-%dT%H:%M:%S', level=logging.DEBUG)

    from PyQt5.QtWidgets import QApplication

    global app # to prevent crash on exit
    app = QApplication(sys.argv)
    w = CMWMain(**kwargs)
    #w.move(0,20)
    w.show()
    app.exec_()
    #del w
    #del app


if __name__ == "__main__":
    calibman(webint=True, loglev='DEBUG')

# EOF
